use std::any::Any;

use hashbrown::HashMap;

use crate::zalsa::Zalsa;
use crate::IngredientIndex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BytesSize {
    pub total: usize,
    pub used: usize,
}

impl BytesSize {
    #[inline]
    pub fn wasted(&self) -> usize {
        self.total.saturating_sub(self.used)
    }

    pub(crate) fn of_hash_map<K, V, S>(map: &HashMap<K, V, S>) -> BytesSize {
        // Taken from https://github.com/Kixiron/size-of/blob/master/src/std_impls.rs, licensed under Apache-2.0/MIT license.
        #[inline]
        const fn capacity_to_buckets(capacity: usize) -> usize {
            if capacity == 0 {
                0
            } else if capacity < 4 {
                4
            } else if capacity < 8 {
                8
            } else {
                (capacity * 8 / 7).next_power_of_two()
            }
        }

        // https://github.com/rust-lang/hashbrown/blob/master/src/raw/generic.rs#L8-L21
        const GROUP_WIDTH: usize = if cfg!(any(
            target_pointer_width = "64",
            target_arch = "aarch64",
            target_arch = "x86_64",
            target_arch = "wasm32",
        )) {
            size_of::<u64>()
        } else {
            size_of::<u32>()
        };

        // https://github.com/rust-lang/hashbrown/blob/2a7c32287247e13680bf874c9a6278aad01fac91/src/raw/mod.rs#L242-L255
        // https://github.com/rust-lang/hashbrown/blob/2a7c32287247e13680bf874c9a6278aad01fac91/src/raw/mod.rs#L1067-L1103
        #[inline]
        pub(crate) const fn calculate_layout_for<T>(buckets: usize) -> usize {
            // FIXME: `max()` isn't a const fn yet
            let align = if align_of::<T>() > GROUP_WIDTH {
                align_of::<T>()
            } else {
                GROUP_WIDTH
            };
            let ctrl_offset = ((size_of::<T>() * buckets) + (align - 1)) & !(align - 1);
            ctrl_offset + buckets + GROUP_WIDTH
        }

        if map.capacity() == 0 {
            return BytesSize { total: 0, used: 0 };
        }

        // Estimate the number of buckets the map contains
        let buckets = capacity_to_buckets(map.capacity());
        // Estimate the layout of the entire table
        let table_layout = calculate_layout_for::<(K, V)>(buckets);
        // Estimate the memory used by `length` elements
        let used_layout = calculate_layout_for::<(K, V)>(map.len());
        BytesSize {
            total: table_layout + size_of::<S>(),
            used: used_layout + size_of::<S>(),
        }
    }
}

#[non_exhaustive]
pub enum DatabaseItem<'a> {
    /// The page's item will be iterated after it, until the next page.
    #[non_exhaustive]
    Page {
        ingredient: IngredientIndex,
        /// The sizes are without the values stored, only the Salsa overhead.
        page_size: BytesSize,
    },
    #[non_exhaustive]
    PageItem {
        fields: &'a (dyn Any + 'static),
        syncs_size: BytesSize,
        memos_size: BytesSize,
    },
    #[non_exhaustive]
    Memo {
        ingredient: IngredientIndex,
        /// This contains `Option<V>`, not `V`!
        value: &'a (dyn Any + 'static),
        /// The sizes are without the values stored, only the Salsa overhead.
        size: usize,
        dependencies_size: BytesSize,
        cycle_heads_size: BytesSize,
        tracked_struct_ids_size: BytesSize,
        // FIXME: Consider accumulated values.
    },
}

pub(crate) fn estimate_database_size(
    zalsa: &mut Zalsa,
    callback: &mut dyn FnMut(DatabaseItem<'_>),
) {
    let memo_ingredient_index_to_ingredient_map = zalsa.memo_ingredient_index_to_ingredient_map();
    let revision = zalsa.current_revision();
    let table = zalsa.table();
    for page in table.all_pages() {
        callback(DatabaseItem::Page {
            ingredient: page.ingredient(),
            page_size: page.size(),
        });

        let memo_types = page.memo_types();
        for slot in page.iter_dyn_slot() {
            // SAFETY: The revision is correct.
            let memos = unsafe { slot.memos(revision) };
            // SAFETY: The revision is correct.
            let syncs = unsafe { slot.syncs(revision) };

            callback(DatabaseItem::PageItem {
                fields: slot.fields(),
                syncs_size: syncs.size(),
                memos_size: memos.size(),
            });

            // SAFETY: The memo types come from the page.
            let memos = unsafe { memo_types.attach_memos(memos) };
            memos.with_memos(|memo_ingredient_index, memo| {
                let ingredient_index = memo_ingredient_index_to_ingredient_map
                    [&(page.ingredient(), memo_ingredient_index)];
                callback(DatabaseItem::Memo {
                    ingredient: ingredient_index,
                    value: memo.value(),
                    size: memo.size_without_value(),
                    dependencies_size: memo.dependencies_size(),
                    cycle_heads_size: memo.cycle_heads_size(),
                    tracked_struct_ids_size: memo.tracked_struct_ids_size(),
                });
            });
        }
    }
}
