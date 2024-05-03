//! Strong typing layer on top of [`GenericListBuilder`]

use crate::{
    builder::{BackendConfig, BuilderConfig},
    types::list::{List, ListSlice},
    ArrayElement,
};
use arrow_array::{
    builder::{ArrayBuilder, GenericListBuilder, ListBuilder},
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
    }

    #[inline]
    fn push(&mut self, s: T::Slice<'_>) {
        self.values().extend_from_slice(s);
        self.append(true)
    }

    fn extend_from_slice(&mut self, s: ListSlice<'_, T>) -> Result<(), ArrowError> {
        let mut values = s.values;

        let total_len = s.lengths.iter().sum::<usize>();
        if total_len != values.len() {
            return Err(ArrowError::InvalidArgumentError(
                "sum of sublist lengths should equate value buffer length".to_string(),
            ));
        }

        for list_len in s.lengths {
            let (sublist, rest) = values.split_at(list_len);
            self.push(sublist);
            values = rest;
        }
        Ok(())
    }
}

// TODO: Also implement for Option<List<T, OffsetSize>> (see below for
//       inspiration)

/*

impl<T: PrimitiveType> TypedBackend<Option<T>> for PrimitiveBuilder<T::Arrow>
where
    // FIXME: Remove this bound once the Rust trait system supports adding the
    //        appropriate bounds on PrimitiveType to let rustc figure out that
    //        T::Value<'_> is just T for primitive types (and thus T::Value must
    //        implement Into<NativeType<T>> per PrimitiveType definition)
    for<'a> T::Value<'a>: Into<NativeType<T>>,
    //
    // FIXME: Remove these bounds once it becomes possible to blanket-impl
    //        ArrayElement for Option<T: PrimitiveType>, making them obvious.
    Option<T>: ArrayElement<ExtendFromSliceResult = Result<(), ArrowError>>,
    for<'a> <Option<T> as ArrayElement>::Value<'a>: Into<Option<T::Value<'a>>>,
    for<'a> <Option<T> as ArrayElement>::Slice<'a>: Into<OptionSlice<'a, T>>,
{
    type Config = ();

    fn new(config: BuilderConfig<Option<T>>) -> Self {
        if let Some(capacity) = config.capacity {
            Self::with_capacity(capacity)
        } else {
            Self::new()
        }
    }

    #[inline]
    fn push(&mut self, v: <Option<T> as ArrayElement>::Value<'_>) {
        let opt: Option<T::Value<'_>> = v.into();
        let opt: Option<NativeType<T>> = opt.map(Into::into);
        self.append_option(opt)
    }

    fn extend_from_slice(
        &mut self,
        slice: <Option<T> as ArrayElement>::Slice<'_>,
    ) -> Result<(), ArrowError> {
        let slice: OptionSlice<T> = slice.into();
        // SAFETY: This transmute is safe for the same reason as above
        let native_values =
            unsafe { std::mem::transmute_copy::<T::Slice<'_>, &[NativeType<T>]>(&slice.values) };
        let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
            self.append_values(native_values, slice.is_valid)
        }));
        res.map_err(|_| {
            ArrowError::InvalidArgumentError("Value and validity lengths must be equal".to_string())
        })
    }
} */

// TODO: Tests
