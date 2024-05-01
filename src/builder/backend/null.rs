//! Strong typing layer on top of [`NullBuilder`]

use super::{Backend, TypedBackend};
use crate::types::primitive::Null;
use arrow_array::builder::NullBuilder;

impl Backend for NullBuilder {
    type ConstructorParameters = ();

    fn new(_params: ()) -> Self {
        Self::new()
    }

    fn with_capacity(_params: (), capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }

    fn capacity(&self) -> usize {
        self.capacity()
    }

    fn extend_with_nulls(&mut self, n: usize) {
        self.append_nulls(n)
    }
}

impl TypedBackend<Null> for NullBuilder {
    #[inline]
    fn push(&mut self, _v: Null) {
        self.append_null()
    }
}
