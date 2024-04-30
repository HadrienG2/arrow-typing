//! Strongly typed abstraction layer over arrow array builders

use crate::{types::Null, ArrayElement, OptionSlice, SliceElement};
use arrow_array::builder::{ArrayBuilder, BooleanBuilder, NullBuilder};
use arrow_schema::ArrowError;
use std::fmt::Debug;

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
impl ExtendFromSlice<bool> for BooleanBuilder {
    fn extend_from_slice(&mut self, s: &[bool]) {
        self.append_slice(s)
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
impl ExtendFromSlice<Option<bool>> for BooleanBuilder {
    fn extend_from_slice(&mut self, slice: OptionSlice<'_, bool>) -> Result<(), ArrowError> {
        self.append_values(slice.values, slice.is_valid)
    }
}

// TODO: Still need to interface...
//
// FixedSizeBinaryBuilder
// FixedSizeListBuilder
// GenericByteBuilder
// GenericByteDictionaryBuilder
// GenericByteRunBuilder
// GenericByteViewBuilder
// GenericListBuilder
// MapBuilder
// PrimitiveBuilder
// PrimitiveDictionaryBuilder
// PrimitiveRunBuilder
// StructBuilder
// UnionBuilder
//
// I should probably start with PrimitiveBuilder, then ListBuilder, then
// StructBuilder, then UnionBuilder, and finish with special cases.

// TODO: Add lower-level backend tests?
