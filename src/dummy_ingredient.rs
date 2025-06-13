use std::sync::Arc;

use crate::function::VerifyResult;
use crate::ingredient::Ingredient;
use crate::plumbing::Location;
use crate::table::memo::MemoTableTypes;
use crate::IngredientIndex;

/// A dummy ingredient, useful when one ingredient wants to have two slot types.
///
/// See [`ZalsaLocal::allocate_with_two_ingredients()`] for an explanation.
///
/// [`ZalsaLocal::allocate_with_two_ingredients()`]: crate::zalsa_local::ZalsaLocal::allocate_with_two_ingredients
#[derive(Debug)]
pub struct DummyIngredient {
    ingredient_index: IngredientIndex,
}

impl DummyIngredient {
    pub fn new(ingredient_index: IngredientIndex) -> DummyIngredient {
        DummyIngredient { ingredient_index }
    }
}

impl Ingredient for DummyIngredient {
    fn debug_name(&self) -> &'static str {
        "DummyIngredient"
    }

    fn location(&self) -> &'static Location {
        const {
            &Location {
                file: file!(),
                line: line!(),
            }
        }
    }

    unsafe fn maybe_changed_after<'db>(
        &'db self,
        _db: &'db dyn crate::Database,
        _input: crate::Id,
        _revision: crate::Revision,
        _cycle_heads: &mut crate::cycle::CycleHeads,
    ) -> VerifyResult {
        unreachable!("DummyIngredient's methods should not get called")
    }

    fn ingredient_index(&self) -> IngredientIndex {
        self.ingredient_index
    }

    fn memo_table_types(&self) -> Arc<MemoTableTypes> {
        unreachable!("DummyIngredient's methods should not get called")
    }
}
