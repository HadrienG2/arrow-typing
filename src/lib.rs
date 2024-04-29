//! A layer on top of [`arrow`](https://docs.rs/arrow) which enables arrow
//! arrays to be built and accessed using idiomatic strongly typed Rust APIs.

pub mod builder;
pub mod types;

use arrow_array::builder::{BooleanBuilder, NullBuilder};

/// Strongly typed data which can be stored as an arrow array element
pub trait ArrayElement: Send + Sync + 'static {
    /// Array builder implementation
    type BuilderBackend: builder::Backend<Self>;

    /// Array element type used for individual element writes and reads
    ///
    /// Most of the type this will just be Self, but sometimes type system
    /// constraints will force us to use a different type. For example, lists of
    /// primitive types T must be written and reads as slices `&[T]`.
    type Value<'a>;
}

// Allow arrow-supported data types to be used in a strongly typed way
impl ArrayElement for () {
    type BuilderBackend = NullBuilder;
    type Value<'a> = Self;
}
//
impl ArrayElement for bool {
    type BuilderBackend = BooleanBuilder;
    type Value<'a> = Self;
}
//
impl ArrayElement for Option<bool> {
    type BuilderBackend = BooleanBuilder;
    type Value<'a> = Self;
}
//
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
// I should probably start with PrimitiveBuilder,

// TODO: Add tests
