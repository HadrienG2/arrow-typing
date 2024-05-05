//! Strong typing layer on top of [`GenericListBuilder`]

use crate::{
    builder::BuilderConfig,
    types::list::{List, ListSlice, OptionListSlice},
    ArrayElement, Slice,
};
use arrow_array::{
    builder::{ArrayBuilder, GenericListBuilder},
    OffsetSizeTrait,
};
use arrow_schema::ArrowError;
use std::fmt::Debug;

use super::{Backend, TypedBackend};

impl<OffsetSize: OffsetSizeTrait, T: ArrayBuilder + Debug> Backend
    for GenericListBuilder<OffsetSize, T>
{
    #[cfg(test)]
    fn capacity_opt(&self) -> Option<usize> {
        None
    }

    fn extend_with_nulls(&mut self, n: usize) {
        for _ in 0..n {
            self.append_null()
        }
    }
}

impl<OffsetSize: OffsetSizeTrait, T: ArrayElement> TypedBackend<List<T, OffsetSize>>
    for GenericListBuilder<OffsetSize, T::BuilderBackend>
{
    type Config = BuilderConfig<T>;

    fn new(config: BuilderConfig<List<T, OffsetSize>>) -> Self {
        let backend = T::BuilderBackend::new(config.backend);
        if let Some(capacity) = config.capacity {
            Self::with_capacity(backend, capacity)
        } else {
            Self::new(backend)
        }
        // FIXME: Manually adjust field configuration to let the user specify
        //        the field name and mark it as non-nullable.
    }

    #[inline]
    fn push(&mut self, s: T::Slice<'_>) {
        self.values().extend_from_slice(s);
        self.append(true)
    }

    fn extend_from_slice(&mut self, s: ListSlice<'_, T>) -> Result<(), ArrowError> {
        if !s.has_consistent_lens() {
            return Err(ArrowError::InvalidArgumentError(
                "sum of sublist lengths should equate value buffer length".to_string(),
            ));
        }
        for sublist in s.iter_cloned() {
            <Self as TypedBackend<List<T, OffsetSize>>>::push(self, sublist);
        }
        Ok(())
    }
}

impl<OffsetSize: OffsetSizeTrait, T: ArrayElement> TypedBackend<Option<List<T, OffsetSize>>>
    for GenericListBuilder<OffsetSize, T::BuilderBackend>
{
    type Config = BuilderConfig<T>;

    fn new(config: BuilderConfig<Option<List<T, OffsetSize>>>) -> Self {
        let backend = T::BuilderBackend::new(config.backend);
        if let Some(capacity) = config.capacity {
            Self::with_capacity(backend, capacity)
        } else {
            Self::new(backend)
        }
        // FIXME: Manually adjust field configuration to let the user specify
        //        the field name and mark it as non-nullable.
    }

    #[inline]
    fn push(&mut self, s: Option<T::Slice<'_>>) {
        if let Some(slice) = s {
            self.values().extend_from_slice(slice);
            self.append(true)
        } else {
            self.append(false)
        }
    }

    fn extend_from_slice(&mut self, s: OptionListSlice<'_, T>) -> Result<(), ArrowError> {
        if !s.has_consistent_lens() {
            return Err(ArrowError::InvalidArgumentError(
                "sum of sublist lengths should equate value buffer length".to_string(),
            ));
        }
        for sublist in s.iter_cloned() {
            <Self as TypedBackend<Option<List<T, OffsetSize>>>>::push(self, sublist);
        }
        Ok(())
    }
}

// TODO: Tests
