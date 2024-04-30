//! A layer on top of [`arrow`](https://docs.rs/arrow) which enables arrow
//! arrays to be built and accessed using strongly typed Rust APIs.

pub mod builder;
pub mod types;

#[cfg(doc)]
use crate::{builder::TypedBuilder, types::Null};
use arrow_array::builder::BooleanBuilder;
use arrow_schema::ArrowError;
use builder::backend::TypedBackend;
use std::fmt::Debug;

/// Strongly typed data which can be stored as an arrow array element
pub trait ArrayElement: Send + Sync + 'static {
    /// Array builder implementation
    type BuilderBackend: builder::backend::TypedBackend<Self>;

    /// Array element type used for individual element writes and reads
    ///
    /// For simple types, this will just be `Self`. But for more complex types,
    /// type system and/or efficiency constraints may force us to use a
    /// different type.
    ///
    /// For example, lists of primitive types T are best read and written as
    /// slices `&[T]`.
    type Value<'a>;
}

/// [`ArrayElement`] which has a null value
///
/// This trait is implemented for both the null element type [`Null`] and
/// options of valid array element types. It enables efficient bulk insertion of
/// null values via [`TypedBuilder::extend_with_nulls()`].
pub trait NullableElement: ArrayElement {}
impl<T: ArrayElement> NullableElement for Option<T> where Option<T>: ArrayElement {}

/// [`ArrayElement`] which can be read or written in bulk using slices
//
// FIXME: The bound I actually want is `ArrayElement<BuilderBackend:
//        ExtendFromSlice<Self>>`, use that once associated type bounds are
//        stable (stabilization PR has landed on nightly at time of writing)
pub trait SliceElement: ArrayElement {
    /// Slice type used for bulk insertion and readout
    ///
    /// For simple types this will just be `&[Self]`, but for more complex
    /// types, efficiency constraints may dictate a different layout.
    ///
    /// For example, nullable primitive types like `Option<u16>` are
    /// bulk-manipulated using [`OptionSlice`] batches. And tuple types like
    /// `(T, U, V)` are bulk-manipulated using `(&[T], &[U], &[V])` batches.
    type Slice<'a>;

    /// Return type of [`TypedBuilder::extend_from_slice()`].
    ///
    /// Bulk insertion always succeeds for simple types. But for complex types
    /// which need composite slice types like `(&[T], &[U])`, bulk insertion can
    /// fail with `ArrowError` if the inner slices have unequal length.
    ///
    /// Accordingly, the return type of `extend_from_slice()` is `()` for
    /// simple slices, and `Result<(), ArrowError>` for composite slices.
    type ExtendFromSliceResult: Debug;
}

/// Alternative to `&[Option<T>]` that is friendlier to columnar storage
pub struct OptionSlice<'a, T: SliceElement> {
    /// Values that may or may not be valid
    pub values: T::Slice<'a>,

    /// Truth that each element of `values` is valid
    pub is_valid: &'a [bool],
}

// Allow arrow-supported standard data types to be used in a strongly typed way
impl ArrayElement for bool {
    type BuilderBackend = BooleanBuilder;
    type Value<'a> = Self;
}
impl SliceElement for bool {
    type Slice<'a> = &'a [Self];
    type ExtendFromSliceResult = ();
}
//
impl<T: ArrayElement> ArrayElement for Option<T>
where
    T::BuilderBackend: TypedBackend<Option<T>>,
{
    type BuilderBackend = T::BuilderBackend;
    type Value<'a> = Option<T::Value<'a>>;
}
impl<T: SliceElement> SliceElement for Option<T>
where
    T::BuilderBackend: TypedBackend<Option<T>>,
{
    type Slice<'a> = OptionSlice<'a, T>;
    type ExtendFromSliceResult = Result<(), ArrowError>;
}
