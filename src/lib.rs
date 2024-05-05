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
pub unsafe trait ArrayElement: Debug + Sized {
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
    type Value<'a>: Clone + Debug + Sized;

    /// Slice type used for bulk insertion and readout
    ///
    /// For simple types this will just be `&[Self::Value]`, but for more
    /// complex types, efficiency constraints may dictate a different layout.
    ///
    /// For example, nullable primitive types like `Option<u16>` are
    /// bulk-manipulated using [`OptionSlice`] batches. And tuple types like
    /// `(T, U, V)` are bulk-manipulated using `(&[T], &[U], &[V])` batches.
    type Slice<'a>: Slice<Value = Self::Value<'a>>;

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

/// A Rust slice, or a generalized version thereof
pub trait Slice: Copy + Clone + Debug + Sized {
    /// Individual slice element or reference thereof yielded by `iter()
    type Value: Debug;

    /// Truth that the inner subslices have a consistent length
    ///
    /// For composite slice types, you should call this and handle the
    /// inconsistent case before calling any other method of this trait.
    fn has_consistent_lens(&self) -> bool;

    /// Number of values that the slice points to
    fn len(&self) -> usize;

    /// Truth that the slice contains no elements
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Values that the slice points to
    fn iter_cloned(&self) -> impl Iterator<Item = Self::Value> + '_;

    /// Split the slice into two subslices at `index`
    ///
    /// # Panics
    ///
    /// Panics if the slice has less than `index` elements.
    fn split_at(&self, index: usize) -> (Self, Self);
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

impl<T: Clone + Debug> Slice for &[T] {
    type Value = T;

    fn has_consistent_lens(&self) -> bool {
        true
    }

    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn iter_cloned(&self) -> impl Iterator<Item = T> + '_ {
        <[T]>::iter(self).cloned()
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        <[T]>::split_at(self, mid)
    }
}

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
        *self
    }
}
//
impl<T: ArrayElement> Copy for OptionSlice<'_, T> {}
//
impl<'a, T: ArrayElement> Slice for OptionSlice<'a, T> {
    type Value = Option<T::Value<'a>>;

    fn has_consistent_lens(&self) -> bool {
        self.values.has_consistent_lens() && self.values.len() == self.is_valid.len()
    }

    fn len(&self) -> usize {
        debug_assert!(self.has_consistent_lens());
        self.is_valid.len()
    }

    fn iter_cloned(&self) -> impl Iterator<Item = Self::Value> + '_ {
        debug_assert!(self.has_consistent_lens());
        self.values
            .iter_cloned()
            .zip(self.is_valid.iter())
            .map(|(value, is_valid)| is_valid.then_some(value))
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        debug_assert!(self.has_consistent_lens());
        let (left_values, right_values) = self.values.split_at(mid);
        let (left_valid, right_valid) = self.is_valid.split_at(mid);
        (
            Self {
                values: left_values,
                is_valid: left_valid,
            },
            Self {
                values: right_values,
                is_valid: right_valid,
            },
        )
    }
}
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
