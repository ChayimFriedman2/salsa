use std::any::{Any, TypeId};
use std::fmt::Debug;
use std::mem;
use std::ptr::{self, NonNull};

use portable_atomic::hint::spin_loop;
use thin_vec::ThinVec;

use crate::sync::atomic::{AtomicPtr, Ordering};
use crate::sync::{OnceLock, RwLock};
use crate::{zalsa::MemoIngredientIndex, zalsa_local::QueryOriginRef};

/// The "memo table" stores the memoized results of tracked function calls.
/// Every tracked function must take a salsa struct as its first argument
/// and memo tables are attached to those salsa structs as auxiliary data.
#[derive(Default)]
pub(crate) struct MemoTable {
    memos: RwLock<ThinVec<MemoEntry>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Either<A, B> {
    Left(A),
    Right(B),
}

/// Represents a `Memo` that has one of two possible types.
///
/// # Safety
///
/// If the `value_type_id()` and the disambiguator match, the value must have the type of
/// the corresponding associated type.
pub unsafe trait AmbiguousMemo {
    /// The `TypeId` of the contained value.
    ///
    /// This can be shared to at most two `Memo` types, distinguished by `MFalse` and `MTrue`.
    /// The important property is that *both* can be stored in a slot. That is, the `MemoEntryType`
    /// only holds the `value_type_id()`, and the disambiguator (true or false) is stored
    /// in the `MemoEntry`.
    fn value_type_id() -> TypeId
    where
        Self: Sized;
    type MFalse: Memo;
    type MTrue: Memo;
}

pub trait Memo: Any + Send + Sync {
    /// Returns the `origin` of this memo
    fn origin(&self) -> QueryOriginRef<'_>;

    /// Returns memory usage information about the memoized value.
    #[cfg(feature = "salsa_unstable")]
    fn memory_usage(&self) -> crate::database::MemoInfo;
}

/// Data for a memoized entry.
/// This is a type-erased `Box<M>`, where `M` is the type of memo associated
/// with that particular ingredient index.
///
/// # Implementation note
///
/// Every entry is associated with some ingredient that has been added to the database.
/// That ingredient has a fixed type of values that it produces etc.
/// Therefore, once a given entry goes from `Empty` to `Full`,
/// the type-id associated with that entry should never change.
///
/// We take advantage of this and use an `AtomicPtr` to store the actual memo.
/// This allows us to store into the memo-entry without acquiring a write-lock.
/// However, using `AtomicPtr` means we cannot use a `Box<dyn Any>` or any other wide pointer.
/// Therefore, we hide the type by transmuting to `DummyMemo`; but we must then be very careful
/// when freeing `MemoEntryData` values to transmute things back. See the `Drop` impl for
/// [`MemoEntry`][] for details.
#[derive(Default)]
struct MemoEntry {
    /// An [`AtomicPtr`][] to a `Box<M>` for the erased memo type `M`
    atomic_memo: AtomicPtr<DummyMemo>,
}

const DISAMBIGUATOR_MASK: usize = 0b1;

/// # Safety
///
/// `ptr` must stay non-null after removing the 0th bit.
#[inline]
unsafe fn unpack_memo_ptr(ptr: NonNull<DummyMemo>) -> (NonNull<DummyMemo>, bool) {
    let ptr = ptr.as_ptr();
    // SAFETY: Our precondition.
    let new_ptr =
        unsafe { NonNull::new_unchecked(ptr.map_addr(|addr| addr & !DISAMBIGUATOR_MASK)) };
    (new_ptr, ptr.addr() & DISAMBIGUATOR_MASK != 0)
}

#[inline]
fn pack_memo_ptr(ptr: NonNull<DummyMemo>, disambiguator: bool) -> NonNull<DummyMemo> {
    // SAFETY: We're ORing bits, it cannot make it null.
    unsafe {
        NonNull::new_unchecked(
            ptr.as_ptr()
                .map_addr(|addr| addr | usize::from(disambiguator)),
        )
    }
}

/// # Safety
///
/// `ptr` must stay non-null after removing the 0th bit. `value_type_id()` must be correct.
#[inline]
unsafe fn unpack_memo_ptr_typed<M: AmbiguousMemo>(
    ptr: NonNull<DummyMemo>,
) -> Either<NonNull<M::MFalse>, NonNull<M::MTrue>> {
    // SAFETY: Our precondition.
    let (new_ptr, disambiguator) = unsafe { unpack_memo_ptr(ptr) };
    match disambiguator {
        false => Either::Left(new_ptr.cast::<M::MFalse>()),
        true => Either::Right(new_ptr.cast::<M::MTrue>()),
    }
}

#[derive(Default)]
pub struct MemoEntryType {
    data: OnceLock<MemoEntryTypeData>,
}

#[derive(Clone, Copy, Debug)]
struct MemoEntryTypeData {
    /// The `type_id` of the erased memo type `M`
    type_id: TypeId,

    /// A type-coercion function for the erased memo type `M`, indexed by `type_id_disambiguator()`.
    to_dyn_fns: [fn(NonNull<DummyMemo>) -> NonNull<dyn Memo>; 2],
}

impl MemoEntryTypeData {
    #[inline]
    fn to_dyn_fn(&self, disambiguator: bool) -> fn(NonNull<DummyMemo>) -> NonNull<dyn Memo> {
        self.to_dyn_fns[usize::from(disambiguator)]
    }
}

impl MemoEntryType {
    fn to_dummy<M: Memo>(memo: NonNull<M>) -> NonNull<DummyMemo> {
        memo.cast()
    }

    const fn to_dyn_fn<M: Memo>() -> fn(NonNull<DummyMemo>) -> NonNull<dyn Memo> {
        let f: fn(NonNull<M>) -> NonNull<dyn Memo> = |x| x;

        // SAFETY: `M: Sized` and `DummyMemo: Sized`, as such they are ABI compatible behind a
        // `NonNull` making it safe to do type erasure.
        unsafe {
            mem::transmute::<
                fn(NonNull<M>) -> NonNull<dyn Memo>,
                fn(NonNull<DummyMemo>) -> NonNull<dyn Memo>,
            >(f)
        }
    }

    #[inline]
    pub fn of<M: AmbiguousMemo>() -> Self {
        const {
            assert!(
                align_of::<M::MFalse>() >= 2,
                "need enough space to encode the disambiguator"
            );
            assert!(
                align_of::<M::MTrue>() >= 2,
                "need enough space to encode the disambiguator"
            );
        };
        Self {
            data: OnceLock::from(MemoEntryTypeData {
                type_id: M::value_type_id(),
                to_dyn_fns: [
                    Self::to_dyn_fn::<M::MFalse>(),
                    Self::to_dyn_fn::<M::MTrue>(),
                ],
            }),
        }
    }

    #[inline]
    fn load(&self) -> Option<&MemoEntryTypeData> {
        self.data.get()
    }
}

/// Dummy placeholder type that we use when erasing the memo type `M` in [`MemoEntryData`][].
#[derive(Debug)]
struct DummyMemo;

impl Memo for DummyMemo {
    fn origin(&self) -> QueryOriginRef<'_> {
        unreachable!("should not get here")
    }

    #[cfg(feature = "salsa_unstable")]
    fn memory_usage(&self) -> crate::database::MemoInfo {
        crate::database::MemoInfo {
            debug_name: "dummy",
            output: crate::database::SlotInfo {
                debug_name: "dummy",
                size_of_metadata: 0,
                size_of_fields: 0,
                memos: Vec::new(),
            },
        }
    }
}

#[derive(Default)]
pub struct MemoTableTypes {
    types: boxcar::Vec<MemoEntryType>,
}

impl MemoTableTypes {
    pub(crate) fn set(
        &self,
        memo_ingredient_index: MemoIngredientIndex,
        memo_type: &MemoEntryType,
    ) {
        let memo_ingredient_index = memo_ingredient_index.as_usize();

        // Try to create our entry if it has not already been created.
        if memo_ingredient_index >= self.types.count() {
            while self.types.push(MemoEntryType::default()) < memo_ingredient_index {}
        }

        loop {
            let Some(memo_entry_type) = self.types.get(memo_ingredient_index) else {
                // It's possible that someone else began pushing to our index but has not
                // completed the entry's initialization yet, as `boxcar` is lock-free. This
                // is extremely unlikely given initialization is just a handful of instructions.
                // Additionally, this function is generally only called on startup, so we can
                // just spin here.
                spin_loop();
                continue;
            };

            memo_entry_type
                .data
                .set(
                    *memo_type.data.get().expect(
                        "cannot provide an empty `MemoEntryType` for `MemoEntryType::set()`",
                    ),
                )
                .expect("memo type should only be set once");
            break;
        }
    }

    /// # Safety
    ///
    /// The types table must be the correct one of `memos`.
    #[inline]
    pub(crate) unsafe fn attach_memos<'a>(
        &'a self,
        memos: &'a MemoTable,
    ) -> MemoTableWithTypes<'a> {
        MemoTableWithTypes { types: self, memos }
    }

    /// # Safety
    ///
    /// The types table must be the correct one of `memos`.
    #[inline]
    pub(crate) unsafe fn attach_memos_mut<'a>(
        &'a self,
        memos: &'a mut MemoTable,
    ) -> MemoTableWithTypesMut<'a> {
        MemoTableWithTypesMut { types: self, memos }
    }
}

pub(crate) struct MemoTableWithTypes<'a> {
    types: &'a MemoTableTypes,
    memos: &'a MemoTable,
}

impl<'a> MemoTableWithTypes<'a> {
    /// # Safety
    ///
    /// The caller needs to make sure to not drop the returned value until no more references into
    /// the database exist as there may be outstanding borrows into the memo contents.
    #[inline]
    pub(crate) unsafe fn insert_false<M: AmbiguousMemo>(
        self,
        memo_ingredient_index: MemoIngredientIndex,
        memo: NonNull<M::MFalse>,
    ) -> Option<Either<NonNull<M::MFalse>, NonNull<M::MTrue>>> {
        let memo = pack_memo_ptr(memo.cast::<DummyMemo>(), false);
        // SAFETY: Our preconditions.
        unsafe { self.insert_impl::<M>(memo_ingredient_index, memo) }
    }
    /// # Safety
    ///
    /// The caller needs to make sure to not drop the returned value until no more references into
    /// the database exist as there may be outstanding borrows into the memo contents.
    #[inline]
    pub(crate) unsafe fn insert_true<M: AmbiguousMemo>(
        self,
        memo_ingredient_index: MemoIngredientIndex,
        memo: NonNull<M::MTrue>,
    ) -> Option<Either<NonNull<M::MFalse>, NonNull<M::MTrue>>> {
        let memo = pack_memo_ptr(memo.cast::<DummyMemo>(), true);
        // SAFETY: Our preconditions.
        unsafe { self.insert_impl::<M>(memo_ingredient_index, memo) }
    }

    /// # Safety
    ///
    /// The caller needs to make sure to not drop the returned value until no more references into
    /// the database exist as there may be outstanding borrows into the memo contents.
    #[inline]
    unsafe fn insert_impl<M: AmbiguousMemo>(
        self,
        memo_ingredient_index: MemoIngredientIndex,
        memo: NonNull<DummyMemo>,
    ) -> Option<Either<NonNull<M::MFalse>, NonNull<M::MTrue>>> {
        // The type must already exist, we insert it when creating the memo ingredient.
        assert_eq!(
            self.types
                .types
                .get(memo_ingredient_index.as_usize())
                .and_then(MemoEntryType::load)?
                .type_id,
            M::value_type_id(),
            "inconsistent type-id for `{memo_ingredient_index:?}`"
        );

        // If the memo slot is already occupied, it must already have the
        // right type info etc, and we only need the read-lock.
        if let Some(MemoEntry { atomic_memo }) = self
            .memos
            .memos
            .read()
            .get(memo_ingredient_index.as_usize())
        {
            let old_memo = atomic_memo.swap(memo.as_ptr(), Ordering::AcqRel);

            let old_memo = NonNull::new(old_memo);

            // SAFETY: `value_type_id()` check asserted above. The pointer points to a valid memo (otherwise
            // it'd be null) so not null.
            return old_memo.map(|old_memo| unsafe { unpack_memo_ptr_typed::<M>(old_memo) });
        }

        // Otherwise we need the write lock.
        self.insert_cold::<M>(memo_ingredient_index, memo)
    }

    #[cold]
    fn insert_cold<M: AmbiguousMemo>(
        self,
        memo_ingredient_index: MemoIngredientIndex,
        memo: NonNull<DummyMemo>,
    ) -> Option<Either<NonNull<M::MFalse>, NonNull<M::MTrue>>> {
        let memo_ingredient_index = memo_ingredient_index.as_usize();
        let mut memos = self.memos.memos.write();

        // Grow the table if needed.
        if memos.len() <= memo_ingredient_index {
            let additional_len = memo_ingredient_index - memos.len() + 1;
            memos.reserve(additional_len);
            while memos.len() <= memo_ingredient_index {
                memos.push(MemoEntry::default());
            }
        }

        let old_entry = mem::replace(
            memos[memo_ingredient_index].atomic_memo.get_mut(),
            MemoEntryType::to_dummy(memo).as_ptr(),
        );

        // SAFETY: `value_type_id()` is asserted in `insert()`. The pointer points to a valid memo (otherwise
        // it'd be null) so not null.
        NonNull::new(old_entry).map(|old_memo| unsafe { unpack_memo_ptr_typed::<M>(old_memo) })
    }

    #[inline]
    pub(crate) fn get<M: AmbiguousMemo>(
        self,
        memo_ingredient_index: MemoIngredientIndex,
    ) -> Option<Either<&'a M::MFalse, &'a M::MTrue>> {
        if let Some(MemoEntry { atomic_memo }) = self
            .memos
            .memos
            .read()
            .get(memo_ingredient_index.as_usize())
        {
            assert_eq!(
                self.types
                    .types
                    .get(memo_ingredient_index.as_usize())
                    .and_then(MemoEntryType::load)?
                    .type_id,
                M::value_type_id(),
                "inconsistent type-id for `{memo_ingredient_index:?}`"
            );
            let memo = NonNull::new(atomic_memo.load(Ordering::Acquire));
            // SAFETY: `value_type_id()` check asserted above. The pointer points to a valid memo (otherwise
            // it'd be null) so not null.
            return memo.map(|old_memo| unsafe {
                match unpack_memo_ptr_typed::<M>(old_memo) {
                    Either::Left(it) => Either::Left(it.as_ref()),
                    Either::Right(it) => Either::Right(it.as_ref()),
                }
            });
        }

        None
    }

    #[cfg(feature = "salsa_unstable")]
    pub(crate) fn memory_usage(&self) -> Vec<crate::database::MemoInfo> {
        let mut memory_usage = Vec::new();
        let memos = self.memos.memos.read();
        for (index, memo) in memos.iter().enumerate() {
            let Some(memo) = NonNull::new(memo.atomic_memo.load(Ordering::Acquire)) else {
                continue;
            };
            // SAFETY: There exists a memo so it's not null.
            let (memo, disambiguator) = unsafe { unpack_memo_ptr(memo) };

            let Some(type_) = self.types.types.get(index).and_then(MemoEntryType::load) else {
                continue;
            };

            // SAFETY: The `TypeId` is asserted in `insert()`.
            let dyn_memo: &dyn Memo = unsafe { type_.to_dyn_fn(disambiguator)(memo).as_ref() };
            memory_usage.push(dyn_memo.memory_usage());
        }

        memory_usage
    }
}

pub(crate) struct MemoTableWithTypesMut<'a> {
    types: &'a MemoTableTypes,
    memos: &'a mut MemoTable,
}

impl MemoTableWithTypesMut<'_> {
    /// Calls `f` on the memo at `memo_ingredient_index`.
    ///
    /// If the memo is not present, `f` is not called.
    pub(crate) fn map_memo<M: AmbiguousMemo>(
        self,
        memo_ingredient_index: MemoIngredientIndex,
        f: impl FnOnce(Either<&mut M::MFalse, &mut M::MTrue>),
    ) {
        let Some(type_) = self
            .types
            .types
            .get(memo_ingredient_index.as_usize())
            .and_then(MemoEntryType::load)
        else {
            return;
        };
        assert_eq!(
            type_.type_id,
            M::value_type_id(),
            "inconsistent type-id for `{memo_ingredient_index:?}`"
        );

        // If the memo slot is already occupied, it must already have the
        // right type info etc, and we only need the read-lock.
        let memos = self.memos.memos.get_mut();
        let Some(MemoEntry { atomic_memo }) = memos.get_mut(memo_ingredient_index.as_usize())
        else {
            return;
        };
        let Some(memo) = NonNull::new(*atomic_memo.get_mut()) else {
            return;
        };

        // SAFETY: `value_type_id()` check asserted above. The pointer points to a valid memo (otherwise
        // it'd be null) so not null.
        f(unsafe {
            match unpack_memo_ptr_typed::<M>(memo) {
                Either::Left(mut it) => Either::Left(it.as_mut()),
                Either::Right(mut it) => Either::Right(it.as_mut()),
            }
        });
    }

    /// To drop an entry, we need its type, so we don't implement `Drop`, and instead have this method.
    ///
    /// Note that calling this multiple times is safe, dropping an uninitialized entry is a no-op.
    ///
    /// # Safety
    ///
    /// The caller needs to make sure to not call this function until no more references into
    /// the database exist as there may be outstanding borrows into the pointer contents.
    #[inline]
    pub unsafe fn drop(&mut self) {
        let types = self.types.types.iter();
        for ((_, type_), memo) in std::iter::zip(types, self.memos.memos.get_mut()) {
            // SAFETY: The types match as per our constructor invariant.
            unsafe { memo.take(type_) };
        }
    }

    /// # Safety
    ///
    /// The caller needs to make sure to not call this function until no more references into
    /// the database exist as there may be outstanding borrows into the pointer contents.
    pub(crate) unsafe fn take_memos(
        &mut self,
        mut f: impl FnMut(MemoIngredientIndex, Box<dyn Memo>),
    ) {
        let memos = self.memos.memos.get_mut();
        memos
            .iter_mut()
            .zip(self.types.types.iter())
            .enumerate()
            .filter_map(|(index, (memo, (_, type_)))| {
                // SAFETY: The types match as per our constructor invariant.
                let memo = unsafe { memo.take(type_)? };
                Some((MemoIngredientIndex::from_usize(index), memo))
            })
            .for_each(|(index, memo)| f(index, memo));
    }
}

impl MemoEntry {
    /// # Safety
    ///
    /// The type must match.
    #[inline]
    unsafe fn take(&mut self, type_: &MemoEntryType) -> Option<Box<dyn Memo>> {
        let memo = NonNull::new(mem::replace(self.atomic_memo.get_mut(), ptr::null_mut()))?;
        let type_ = type_.load()?;
        // SAFETY: We store an actual memo (otherwise `self.atomic_memo` would be null) of this type (our precondition).
        let (memo, disambiguator) = unsafe { unpack_memo_ptr(memo) };
        // SAFETY: Our preconditions.
        Some(unsafe { Box::from_raw(type_.to_dyn_fn(disambiguator)(memo).as_ptr()) })
    }
}

impl Drop for DummyMemo {
    fn drop(&mut self) {
        unreachable!("should never get here")
    }
}

impl std::fmt::Debug for MemoTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoTable").finish_non_exhaustive()
    }
}
