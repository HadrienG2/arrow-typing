//! Strong typing layer on top of Arrow builders

mod bool;
mod null;
mod primitive;

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

use crate::{ArrayElement, SliceElement};
use arrow_array::builder::ArrayBuilder;
use std::fmt::Debug;

use super::BuilderConfig;

/// Arrow builder that can accept strongly typed entries of type `T`
pub trait TypedBackend<T: ArrayElement + ?Sized>: Backend {
    /// Configuration needed to construct a builder backend for this type
    type Config;

    /// Create a new builder backend
    fn new(config: BuilderConfig<T>) -> Self;

    /// Append a single element into the builder
    fn push(&mut self, v: T::Value<'_>);
}

/// Subset of `TypedBackend<T>` functionality that does not depend on `T`
pub trait Backend: ArrayBuilder + Debug {
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

/// Access the current null buffer as a slice
pub trait ValiditySlice: Backend {
    /// Returns the current null buffer as a slice
    fn validity_slice(&self) -> Option<&[u8]>;
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
