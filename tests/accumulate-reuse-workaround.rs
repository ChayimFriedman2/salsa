//! Demonstrates the workaround of wrapping calls to
//! `accumulated` in a tracked function to get better
//! reuse.

mod common;
use common::{HasLogger, Logger};

use expect_test::expect;
use salsa::{Accumulator, Setter};
use test_log::test;

#[salsa::db]
trait Db: salsa::Database + HasLogger {}

#[salsa::input]
struct List {
    value: u32,
    next: Option<List>,
}

#[salsa::accumulator]
#[derive(Clone, Copy, Debug)]
struct Integers(u32);

#[salsa::tracked]
fn compute(db: &dyn Db, input: List) -> u32 {
    db.push_log(format!("compute({:?})", input,));

    // always pushes 0
    Integers(0).accumulate(db);

    let result = if let Some(next) = input.next(db) {
        let next_integers = accumulated(db, next);
        let v = input.value(db) + next_integers.iter().sum::<u32>();
        v
    } else {
        input.value(db)
    };

    // return value changes
    result
}

#[salsa::tracked(return_ref)]
fn accumulated(db: &dyn Db, input: List) -> Vec<u32> {
    db.push_log(format!("accumulated({:?})", input,));
    compute::accumulated::<Integers>(db, input)
        .into_iter()
        .map(|a| a.0)
        .collect()
}

#[salsa::db]
#[derive(Default)]
struct Database {
    storage: salsa::Storage<Self>,
    logger: Logger,
}

#[salsa::db]
impl salsa::Database for Database {
    fn salsa_event(&self, _event: salsa::Event) {}
}

#[salsa::db]
impl Db for Database {}

impl HasLogger for Database {
    fn logger(&self) -> &Logger {
        &self.logger
    }
}

#[test]
fn test1() {
    let mut db = Database::default();

    let l1 = List::new(&db, 1, None);
    let l2 = List::new(&db, 2, Some(l1));

    assert_eq!(compute(&db, l2), 2);
    db.assert_logs(expect![[r#"
        [
            "compute(List { [salsa id]: 1 })",
            "accumulated(List { [salsa id]: 0 })",
            "compute(List { [salsa id]: 0 })",
        ]"#]]);

    // When we mutate `l1`, we should re-execute `compute` for `l1`,
    // and we re-execute accumulated for `l1`, but we do NOT re-execute
    // `compute` for `l2`.
    l1.set_value(&mut db).to(2);
    assert_eq!(compute(&db, l2), 2);
    db.assert_logs(expect![[r#"
        [
            "accumulated(List { [salsa id]: 0 })",
            "compute(List { [salsa id]: 0 })",
        ]"#]]);
}
