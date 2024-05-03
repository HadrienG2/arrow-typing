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
pub unsafe trait ArrayElement: Debug + Send + Sized + Sync + 'static {
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
    type Slice<'a>: Copy + Clone + Debug;

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
#[derive(Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OptionSlice<'a, T: ArrayElement> {
    /// Values that may or may not be valid
    pub values: T::Slice<'a>,

    /// Truth that each element of `values` is valid
    pub is_valid: &'a [bool],
}
//
impl<T: ArrayElement> Clone for OptionSlice<'_, T> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            is_valid: self.is_valid,
        }
    }
}
//
impl<T: ArrayElement> Copy for OptionSlice<'_, T> {}
//
// NOTE: I tried to make this blanket-impl'd for Option<T> where
//       T::BuilderBackend: TypedBackend<Option<T>>, but this caused
//       problems down the line where backends were not recognized
//       by the trait solver as implementing TypedBackend<Option<T>>
//       because Option<T> did not implement ArrayElement. Let's
//       keep this macrofied for now.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_option_element {
    ($t:ty) => {
        // SAFETY: Option is not a primitive type and is therefore not
        //         affected by the safety precondition of ArrayElement
        unsafe impl ArrayElement for Option<$t> {
            type BuilderBackend = <$t as ArrayElement>::BuilderBackend;
            type Value<'a> = Option<<$t as ArrayElement>::Value<'a>>;
            type Slice<'a> = $crate::OptionSlice<'a, $t>;
            type ExtendFromSliceResult = Result<(), arrow_schema::ArrowError>;
        }
    };
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
