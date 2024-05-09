//! Strong typing layer on top of Arrow builders

pub mod bool;
pub mod list;
pub mod null;
pub mod primitive;

// FIXME: Still need to interface remaining API of GenericListBuilder and
//        PrimitiveBuilder and then...
//
// - FixedSizeBinaryBuilder
// - FixedSizeListBuilder
// - GenericByteBuilder
// - GenericByteDictionaryBuilder
// - GenericByteRunBuilder
// - GenericByteViewBuilder
// - MapBuilder
// - PrimitiveDictionaryBuilder
// - PrimitiveRunBuilder
// - StructBuilder
// - UnionBuilder
//
// I should probably start with StructBuilder, then UnionBuilder, and finish
// with special cases.

use super::BuilderConfig;
use crate::element::ArrayElement;
use arrow_array::builder::ArrayBuilder;
use arrow_schema::Field;
use std::fmt::Debug;

/// Arrow builder that can accept strongly typed entries of type `T`
pub trait TypedBackend<T: ArrayElement>: Backend {
    /// Extra configuration requested by the new/with_capacity constructor
    type ExtraConfig: Debug + PartialEq;

    /// Enum of alternate constructors/parameters besides new/with_capacity
    type AlternateConfig: Debug + PartialEq + Capacity;

    /// Make a structured array [`Field`] of this type with a certain name
    fn make_field(config: &BuilderConfig<T>, name: String) -> Field;

    /// Create a new builder backend for an array of this type
    fn new(config: BuilderConfig<T>) -> Self;

    /// Append a single element into the builder
    ///
    /// Implementors should almost always make this operation `#[inline]` to
    /// allow for cross-crate inlining.
    fn push(&mut self, v: T::Value<'_>);

    /// Append values into the builder in bulk
    fn extend_from_slice(&mut self, s: T::Slice<'_>) -> T::ExtendFromSliceResult;
}

/// Best-effort buffer builder capacity query
///
/// When an Arrow array builder provides a way to query the capacity, or it can
/// be emulated on our side, this trait can be used to access it. The trait is
/// also implemented for the `AlternateConfig` configuration methods, where
/// available, in which case it behaves as a prediction of minimal final builder
/// capacity once a builder is built using that configuration.
///
/// In the case of types that are internally stored as multiple columnar
/// buffers, like structs or unions, a lower bound on the capacity of all
/// underlying columns is returned.
///
/// In the case of arrays of lists, capacity is to be understood as the
/// number of sublists that the array can hold, not the cumulative number of
/// elements across all sublists. But this is not a concern yet since list
/// builders do not currently expose capacity anyway.
pub trait Capacity {
    /// Number of elements the builder can hold without reallocating
    fn capacity(&self) -> usize;
}

/// Marker type denoting absence of alternate configuration methods
///
/// Used as the `TypedBackend::AlternateConfig` type when only
/// `new()`/`with_capacity()` construction is available.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum NoAlternateConfig {}
//
impl Capacity for NoAlternateConfig {
    fn capacity(&self) -> usize {
        unreachable!()
    }
}

/// Subset of `TypedBackend<T>` functionality that does not depend on `T`
pub trait Backend: ArrayBuilder + Debug {
    /// Backend capacity if available, otherwise None
    #[cfg(test)]
    fn capacity_opt(&self) -> Option<usize>;

    /// Efficiently append `n` null values into the builder
    fn extend_with_nulls(&mut self, n: usize);
}

/// Access the current null buffer as a slice
pub trait ValiditySlice: Backend {
    /// Returns the current null buffer as a slice
    fn validity_slice(&self) -> Option<&[u8]>;
}
