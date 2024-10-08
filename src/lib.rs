//! A layer on top of [`arrow`](https://docs.rs/arrow) which enables arrow
//! arrays to be built and accessed using strongly typed Rust APIs.

pub mod builder;
pub mod types;
pub mod validity;

use crate::types::primitive::Null;
#[cfg(doc)]
use crate::types::primitive::PrimitiveType;
#[cfg(doc)]
use arrow_schema::ArrowError;
use std::fmt::Debug;

pub use builder::TypedBuilder;

/// Strongly typed data which can be stored as an Arrow array element
///
/// # Safety
///
/// If this trait is implemented on a [primitive type](PrimitiveType), then the
/// `Slice` associated type **must** be set to `&[Self]`.
pub unsafe trait ArrayElement: Debug + Send + Sync + 'static {
    /// Array builder implementation
    #[doc(hidden)]
    type BuilderBackend: builder::backend::TypedBackend<Self>;

    /// Array element type used for individual element writes and reads
    ///
    /// For simple types, this will just be `Self`. But for more complex types,
    /// type system, ergonomics and efficiency constraints may force us to use a
    /// different type.
    ///
    /// For example, lists of primitive types T are best read and written as
    /// slices `&[T]`.
    type Value<'a>: Debug;

    /// Slice type used for bulk insertion and readout
    ///
    /// For simple types this will just be `&[Self::Value]`, but for more
    /// complex types, efficiency constraints may dictate a different layout.
    ///
    /// For example, nullable primitive types like `Option<u16>` are
    /// bulk-manipulated using [`OptionSlice`] batches. And tuple types like
    /// `(T, U, V)` are bulk-manipulated using `(&[T], &[U], &[V])` batches.
    type Slice<'a>: Debug;

    /// Return type of [`TypedBuilder::extend_from_slice()`].
    ///
    /// Bulk insertion always succeeds for simple types. But for complex types
    /// which need composite slice types like `(&[T], &[U])`, bulk insertion can
    /// fail with [`ArrowError`] if the inner slices have unequal length.
    ///
    /// Accordingly, the return type of `extend_from_slice()` is `()` for
    /// simple slices, and `Result<(), ArrowError>` for composite slices.
    type ExtendFromSliceResult: Debug;
}

/// [`ArrayElement`] which has a null value
///
/// This trait is implemented for both the null element type [`Null`] and
/// options of valid array element types. It enables efficient bulk insertion of
/// null values via [`TypedBuilder::extend_with_nulls()`].
pub trait NullableElement: ArrayElement {}
//
impl NullableElement for Null {}
//
impl<T: ArrayElement> NullableElement for Option<T> where Option<T>: ArrayElement {}

/// Columnar alternative to `&[Option<T>]`
#[derive(Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OptionSlice<'a, T: ArrayElement> {
    /// Values that may or may not be valid
    pub values: T::Slice<'a>,

    /// Truth that each element of `values` is valid
    pub is_valid: &'a [bool],
}
//
impl<'a, T: ArrayElement> Clone for OptionSlice<'a, T>
where
    T::Slice<'a>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            is_valid: self.is_valid,
        }
    }
}

/// Shared test utilities
#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    /// Maximum array length/capacity used in unit tests
    pub const MAX_CAPACITY: usize = 256;

    /// Generate a capacity between 0 and MAX_CAPACITY
    pub fn length_or_capacity() -> impl Strategy<Value = usize> {
        0..=MAX_CAPACITY
    }
}
