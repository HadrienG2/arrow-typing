//! Strongly typed abstraction layer over arrow array builders

use crate::{
    types::primitive::{AsArrowPrimitive, NativeType, Null},
    ArrayElement, OptionSlice, SliceElement,
};
use arrow_array::{
    builder::{ArrayBuilder, BooleanBuilder, NullBuilder, PrimitiveBuilder},
    types::ArrowPrimitiveType,
};
use arrow_schema::ArrowError;
use std::{fmt::Debug, panic::AssertUnwindSafe};

// === Arrow builder abstraction layer ===

/// Arrow builder that can accept strongly typed entries of type `T`
pub trait TypedBackend<T: ArrayElement + ?Sized>: Backend {
    /// Append a single element into the builder
    fn push(&mut self, v: T::Value<'_>);
}

/// Bulk-insertion of [`SliceElement`]s into corresponding arrow arrays
//
// --- Implementation notes ---
//
// In the interest of reducing the number of traits that maintainers of this
// code need to juggle with, this should arguably be an optional method of
// [`TypedBackend`] with a `where T: SliceElement` bound.
//
// Alas rustc's trait solver is not yet ready for this because until
// https://github.com/rust-lang/rust/issues/48214 is resolved, it will results
// in `TypedBackend` being unimplementable when T **does not** implement
// `SliceElement`. Therefore, a separate trait is needed for now.
pub trait ExtendFromSlice<T: SliceElement + ?Sized>: TypedBackend<T> {
    /// Append values into the builder in bulk
    fn extend_from_slice(&mut self, s: T::Slice<'_>) -> T::ExtendFromSliceResult;
}

/// Subset of `TypedBackend<T>` functionality that does not depend on `T`
pub trait Backend: ArrayBuilder + Debug {
    /// Constructor parameters other than inner array builders
    type ConstructorParameters;

    /// Create a new builder with no underlying buffer allocation
    fn new(params: Self::ConstructorParameters) -> Self;

    /// Create a new builder with space for `capacity` elements
    fn with_capacity(params: Self::ConstructorParameters, capacity: usize) -> Self;

    /// Number of elements the array can hold without reallocating
    ///
    /// In the case of types that are internally stored as multiple columnar
    /// buffers, like structs or unions, a lower bound on the capacity of all
    /// underlying columns is returned.
    ///
    /// In the case of arrays of lists, capacity is to be understood as the
    /// number of sublists that the array can hold, not the cumulative number of
    /// elements across all sublists.
    fn capacity(&self) -> usize;

    /// Efficiently append `n` null values into the builder
    fn extend_with_nulls(&mut self, n: usize);
}

// === Implementation of the abstraction layer for arrow builders ===

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
//
impl TypedBackend<Null> for NullBuilder {
    #[inline]
    fn push(&mut self, _v: Null) {
        self.append_null()
    }
}

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
//
impl TypedBackend<bool> for BooleanBuilder {
    #[inline]
    fn push(&mut self, v: bool) {
        self.append_value(v)
    }
}
//
impl TypedBackend<Option<bool>> for BooleanBuilder {
    #[inline]
    fn push(&mut self, v: Option<bool>) {
        self.append_option(v)
    }
}
//
impl ExtendFromSlice<bool> for BooleanBuilder {
    fn extend_from_slice(&mut self, s: &[bool]) {
        self.append_slice(s)
    }
}
//
impl ExtendFromSlice<Option<bool>> for BooleanBuilder {
    fn extend_from_slice(&mut self, slice: OptionSlice<'_, bool>) -> Result<(), ArrowError> {
        self.append_values(slice.values, slice.is_valid)
    }
}

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
//
impl<T: AsArrowPrimitive> TypedBackend<T> for PrimitiveBuilder<T::ArrowPrimitive>
where
    // FIXME: Remove this bound once the Rust trait system supports adding the
    //        appropriate bounds on PrimitiveType to let rustc figure out that
    //        T::Value<'_> is just T for primitive types.
    for<'a> T::Value<'a>: AsArrowPrimitive + From<NativeType<T>> + Into<NativeType<T>>,
{
    #[inline]
    fn push(&mut self, v: T::Value<'_>) {
        self.append_value(v.into())
    }
}
//
impl<T: AsArrowPrimitive> TypedBackend<Option<T>> for PrimitiveBuilder<T::ArrowPrimitive>
where
    // FIXME: Remove these bounds once the Rust trait system supports adding the
    //        appropriate bounds on PrimitiveType to let rustc figure out that
    //        T::Value<'_> is just T for primitive types.
    for<'a> T::Value<'a>: AsArrowPrimitive + From<NativeType<T>> + Into<NativeType<T>>,
    <T as ArrayElement>::BuilderBackend: TypedBackend<Option<T>>,
{
    #[inline]
    fn push(&mut self, v: Option<T::Value<'_>>) {
        self.append_option(v.map(Into::into))
    }
}
//
impl<T: AsArrowPrimitive<ExtendFromSliceResult = ()>> ExtendFromSlice<T>
    for PrimitiveBuilder<T::ArrowPrimitive>
where
    // FIXME: Remove these bounds once the Rust trait system supports adding the
    //        appropriate bounds on PrimitiveType to let rustc figure out that
    //        T::Value<'_> is just T for primitive types.
    for<'a> T::Value<'a>: AsArrowPrimitive + From<NativeType<T>> + Into<NativeType<T>>,
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
//
impl<T: AsArrowPrimitive<ExtendFromSliceResult = Result<(), ArrowError>>> ExtendFromSlice<Option<T>>
    for PrimitiveBuilder<T::ArrowPrimitive>
where
    // FIXME: Remove these bounds once the Rust trait system supports adding the
    //        appropriate bounds on PrimitiveType to let rustc figure out that
    //        T::Value<'_> is just T for primitive types.
    for<'a> T::Value<'a>: AsArrowPrimitive + From<NativeType<T>> + Into<NativeType<T>>,
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

// TODO: Still need to interface remaining API of PrimitiveBuilder and then...
//
// - FixedSizeBinaryBuilder
// - FixedSizeListBuilder
// - GenericByteBuilder
// - GenericByteDictionaryBuilder
// - GenericByteRunBuilder
// - GenericByteViewBuilder
// - GenericListBuilder
// - MapBuilder
// - PrimitiveDictionaryBuilder
// - PrimitiveRunBuilder
// - StructBuilder
// - UnionBuilder
//
// I should probably start with PrimitiveBuilder, then ListBuilder, then
// StructBuilder, then UnionBuilder, and finish with special cases.

// TODO: Add lower-level backend tests?
