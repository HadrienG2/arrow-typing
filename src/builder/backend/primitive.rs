//! Strong typing layer on top of [`PrimitiveBuilder`]

use super::{Backend, ExtendFromSlice, TypedBackend, ValiditySlice};
use crate::{
    types::primitive::{NativeType, PrimitiveType},
    ArrayElement, OptionSlice,
};
use arrow_array::{builder::PrimitiveBuilder, types::ArrowPrimitiveType};
use arrow_schema::ArrowError;
use std::{fmt::Debug, panic::AssertUnwindSafe};

impl<T: ArrowPrimitiveType + Debug> Backend for PrimitiveBuilder<T> {
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

impl<T: ArrowPrimitiveType + Debug> ValiditySlice for PrimitiveBuilder<T> {
    fn validity_slice(&self) -> Option<&[u8]> {
        self.validity_slice()
    }
}

impl<T: PrimitiveType> TypedBackend<T> for PrimitiveBuilder<T::Arrow>
where
    // FIXME: Remove this bound once the Rust trait system supports adding the
    //        appropriate bounds on PrimitiveType to let rustc figure out that
    //        T::Value<'_> is just T for primitive types.
    for<'a> T::Value<'a>: PrimitiveType + From<NativeType<T>> + Into<NativeType<T>>,
{
    #[inline]
    fn push(&mut self, v: T::Value<'_>) {
        self.append_value(v.into())
    }
}

impl<T: PrimitiveType> TypedBackend<Option<T>> for PrimitiveBuilder<T::Arrow>
where
    // FIXME: Remove these bounds once the Rust trait system supports adding the
    //        appropriate bounds on PrimitiveType to let rustc figure out that
    //        T::Value<'_> is just T for primitive types.
    for<'a> T::Value<'a>: PrimitiveType + From<NativeType<T>> + Into<NativeType<T>>,
    <T as ArrayElement>::BuilderBackend: TypedBackend<Option<T>>,
{
    #[inline]
    fn push(&mut self, v: Option<T::Value<'_>>) {
        self.append_option(v.map(Into::into))
    }
}

impl<T: PrimitiveType<ExtendFromSliceResult = ()>> ExtendFromSlice<T> for PrimitiveBuilder<T::Arrow>
where
    // FIXME: Remove these bounds once the Rust trait system supports adding the
    //        appropriate bounds on PrimitiveType to let rustc figure out that
    //        T::Value<'_> is just T for primitive types.
    for<'a> T::Value<'a>: PrimitiveType + From<NativeType<T>> + Into<NativeType<T>>,
{
    fn extend_from_slice(&mut self, s: T::Slice<'_>) {
        // SAFETY: This transmute is safe because...
        //         - T::Slice is &[T] for all primitive types
        //         - Primitive types are repr(transparent) wrappers over the
        //           corresponding Arrow native types, so it is safe to
        //           transmute &[T] into &[NativeType<T>].
        let native_slice =
            unsafe { std::mem::transmute_copy::<T::Slice<'_>, &[NativeType<T>]>(&s) };
        self.append_slice(native_slice)
    }
}

impl<T: PrimitiveType<ExtendFromSliceResult = Result<(), ArrowError>>> ExtendFromSlice<Option<T>>
    for PrimitiveBuilder<T::Arrow>
where
    // FIXME: Remove these bounds once the Rust trait system supports adding the
    //        appropriate bounds on PrimitiveType to let rustc figure out that
    //        T::Value<'_> is just T for primitive types.
    for<'a> T::Value<'a>: PrimitiveType + From<NativeType<T>> + Into<NativeType<T>>,
    <T as ArrayElement>::BuilderBackend: TypedBackend<Option<T>>,
{
    fn extend_from_slice(&mut self, slice: OptionSlice<'_, T>) -> Result<(), ArrowError> {
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
}
