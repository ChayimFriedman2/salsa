#![cfg(feature = "inventory")]

//! Test that a `tracked` fn on a `salsa::input`
//! compiles and executes successfully.

use std::convert::identity;

use salsa::{Database, HashEqLike, Lookup};
use test_log::test;

#[test]
fn interning_returns_equal_keys_for_equal_data() {
    let db = salsa::DatabaseImpl::new();
    let s1 = InternedString::new(&db, "Hello, ".to_string(), identity);
    let s2 = InternedString::new(&db, "World, ".to_string(), |_| s1);
    let s1_2 = InternedString::new(&db, "Hello, ", identity);
    let s2_2 = InternedString::new(&db, "World, ", |_| s2);
    assert_eq!(s1, s1_2);
    assert_eq!(s2, s2_2);
}

#[salsa::interned(debug, constructor = new_impl)]
struct InternedString<'db> {
    data: String,
    #[no_eq]
    other: InternedString<'db>,
}

impl<'db> InternedString<'db> {
    pub fn new(
        db: &'db dyn Database,
        data: impl HashEqLike<String> + Lookup<String>,
        other: impl FnOnce(InternedString<'db>) -> InternedString<'db>,
    ) -> Self {
        struct OtherLookup<F>(F);

        impl<'db, F> Lookup<InternedString<'db>> for OtherLookup<F>
        where
            F: FnOnce(InternedString<'db>) -> InternedString<'db>,
        {
            fn into_owned(self, id: salsa::Id) -> InternedString<'db> {
                (self.0)(salsa::plumbing::FromId::from_id(id))
            }
        }

        impl<'db, F> HashEqLike<InternedString<'db>> for OtherLookup<F> {
            fn hash<H: std::hash::Hasher>(&self, _h: &mut H) {}

            fn eq(&self, _data: &InternedString<'db>) -> bool {
                true
            }
        }

        Self::new_impl(db, data, OtherLookup(other))
    }
}
