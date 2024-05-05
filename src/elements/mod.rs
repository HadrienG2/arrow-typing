//! Array elements and slices thereof

pub mod list;
pub mod primitive;

use crate::{builder::backend::TypedBackend, elements::primitive::Null};
#[cfg(doc)]
use crate::{builder::TypedBuilder, elements::primitive::PrimitiveType};
#[cfg(doc)]
use arrow_schema::ArrowError;
use std::fmt::Debug;

/// Arrow array element
///
/// # Safety
///
/// When this trait is implemented on a type that also implements the
/// [`PrimitiveType`] trait, its `Slice` generic associated type must be set to
/// `&[Self]`.
pub unsafe trait ArrayElement: Debug + Sized {
    /// Array builder implementation
    #[doc(hidden)]
    type BuilderBackend: TypedBackend<Self>;

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

/// A Rust slice `&[T]` or a columnar generalization thereof
pub trait Slice: Copy + Clone + Debug + Sized {
    /// Individual slice element, as an owned value
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

    /// Copies of the values that the slice points to
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
