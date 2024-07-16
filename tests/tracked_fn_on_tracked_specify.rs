//! Test that a `tracked` fn on a `salsa::input`
//! compiles and executes successfully.
#![allow(warnings)]

#[salsa::jar(db = Db)]
struct Jar(MyInput, MyTracked<'_>, tracked_fn, tracked_fn_extra);

trait Db: salsa::DbWithJar<Jar> {}

#[salsa::input]
struct MyInput {
    field: u32,
}

#[salsa::tracked]
struct MyTracked<'db> {
    field: u32,
}

#[salsa::tracked]
fn tracked_fn<'db>(db: &'db dyn Db, input: MyInput) -> MyTracked<'db> {
    let t = MyTracked::new(db, input.field(db) * 2);
    if input.field(db) != 0 {
        tracked_fn_extra::specify(db, t, 2222);
    }
    t
}

#[salsa::tracked(jar = Jar, specify)]
fn tracked_fn_extra<'db>(_db: &'db dyn Db, _input: MyTracked<'db>) -> u32 {
    0
}

#[salsa::db(Jar)]
#[derive(Default)]
struct Database {
    storage: salsa::Storage<Self>,
}

impl salsa::Database for Database {}

impl Db for Database {}

#[test]
fn execute_when_specified() {
    let mut db = Database::default();
    let input = MyInput::new(&db, 22);
    let tracked = tracked_fn(&db, input);
    assert_eq!(tracked.field(&db), 44);
    assert_eq!(tracked_fn_extra(&db, tracked), 2222);
}

#[test]
fn execute_when_not_specified() {
    let mut db = Database::default();
    let input = MyInput::new(&db, 0);
    let tracked = tracked_fn(&db, input);
    assert_eq!(tracked.field(&db), 0);
    assert_eq!(tracked_fn_extra(&db, tracked), 0);
}
