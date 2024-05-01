//! Strong typing layer on top of [`BooleanBuilder`]

use super::{Backend, ExtendFromSlice, TypedBackend};
use crate::OptionSlice;
use arrow_array::builder::BooleanBuilder;
use arrow_schema::ArrowError;

impl Backend for BooleanBuilder {
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

impl TypedBackend<bool> for BooleanBuilder {
    #[inline]
    fn push(&mut self, v: bool) {
        self.append_value(v)
    }
}

impl TypedBackend<Option<bool>> for BooleanBuilder {
    #[inline]
    fn push(&mut self, v: Option<bool>) {
        self.append_option(v)
    }
}

impl ExtendFromSlice<bool> for BooleanBuilder {
    fn extend_from_slice(&mut self, s: &[bool]) {
        self.append_slice(s)
    }
}

impl ExtendFromSlice<Option<bool>> for BooleanBuilder {
    fn extend_from_slice(&mut self, slice: OptionSlice<'_, bool>) -> Result<(), ArrowError> {
        self.append_values(slice.values, slice.is_valid)
    }
}
