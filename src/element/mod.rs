//! Array elements and slices thereof

pub mod list;
pub mod primitive;

use crate::{builder::backend::TypedBackend, element::primitive::Null};
#[cfg(doc)]
use crate::{builder::TypedBuilder, element::primitive::PrimitiveType, validity::ValiditySlice};
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

    /// Array element type used when writing to the array
    ///
    /// For simple types, this will just be `Self`. But for more complex types,
    /// ergonomics and efficiency constraints may force us to use a different
    /// type.
    ///
    /// For example, lists whose items are of a primitive type are best read and
    /// written as slices `&[T]`.
    type WriteValue<'a>: Clone + Debug + Sized;

    /// Array element type used when reading from an array
    ///
    /// For simple types, this will be the same as `WriteValue`. But for more
    /// complex types, especially those involving lists, the two types can
    /// differ.
    ///
    /// The reason why this happens is that in-place reads from Arrow arrays
    /// must stay close to the underlying Arrow storage format to be efficient,
    /// even if this comes at the expense of reduced ergonomics and
    /// compatibility with Rust idoms. For example...
    ///
    /// - [Slices of optional items](OptionSlice) must be implemented using the
    ///   bit-packed [`ValiditySlice`] null buffer format, instead of the more
    ///   idiomatic unpacked `&[bool]` slices that are used for writing.
    /// - Slices of lists must be implemented using a signed offset buffer to
    ///   encode the position and size of inner sublists, instead of an array of
    ///   sublist lengths as done for writes.
    ///
    /// In constrast, writes are (relatively) free to use more idiomatic data
    /// structures that are easier for a user to construct.
    ///
    /// Ultimately, `WriteValue` and `ReadValue` will always be two
    /// implementations of the same logical concept (e.g. if `WriteValue` is a
    /// slice of booleans, `ReadValue` will be a slice of booleans as well). It
    /// just happens that the implementations may sometimes differ.
    type ReadValue<'a>: Clone + Debug + Sized;

    /// Slice type used for bulk insertion into an array
    ///
    /// For simple types this will just be `&[Self::WriteValue]`. But for more
    /// complex types, efficiency constraints may dictate a different layout.
    ///
    /// For example, nullable primitive types like `Option<u16>` are
    /// bulk-manipulated using [`OptionSlice`] batches. And tuple types like
    /// `(T, U, V)` are bulk-manipulated using `(&[T], &[U], &[V])` batches.
    type WriteSlice<'a>: Slice<Value = Self::WriteValue<'a>>;

    /// Slice type used for bulk readout from an array
    ///
    /// `ReadSlice` is to `WriteSlice` what `ReadValue` is to `WriteValue`: an
    /// alternate implementation of the same logical concept which is optimized
    /// for efficient Arrow data readout, instead of usage ergonomics.
    type ReadSlice<'a>: Slice<Value = Self::ReadValue<'a>>;

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
    ///
    /// Implementations of this method should normally be `#[inline]`.
    fn len(&self) -> usize;

    /// Truth that the slice contains no elements
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Value of the `index`-th element, if in bounds
    #[inline]
    fn get_cloned(&self, index: usize) -> Option<Self::Value> {
        (index < self.len()).then(|| unsafe { self.get_cloned_unchecked(index) })
    }

    /// Value of the `index`-th validity bit, without bounds checking
    ///
    /// For a safe alternative see [`get_cloned`](Self::get_cloned).
    ///
    /// Implementations of this method should normally be `#[inline]`.
    ///
    /// # Safety
    ///
    /// `index` must be in bounds or undefined behavior will ensue.
    unsafe fn get_cloned_unchecked(&self, index: usize) -> Self::Value;

    /// Value of the `index`-th, with panic-based bounds checking
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    fn at(&self, index: usize) -> Self::Value {
        self.get_cloned(index).expect("index is out of bounds")
    }

    /// Copies of the values that the slice points to
    fn iter_cloned(&self) -> impl Iterator<Item = Self::Value> + '_;

    /// Split the slice into two subslices at `index`
    ///
    /// # Panics
    ///
    /// Panics if the slice has less than `index` elements.
    fn split_at(&self, mid: usize) -> (Self, Self);
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

// Reference implementation of the Slice trait... on slices ;)
impl<T: Clone + Debug> Slice for &[T] {
    type Value = T;

    fn has_consistent_lens(&self) -> bool {
        true
    }

    #[inline]
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, index: usize) -> T {
        unsafe { <[T]>::get_unchecked(self, index).clone() }
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
pub struct OptionSlice<'a, T, Validity = &'a [bool]>
where
    T: ArrayElement,
    Validity: Slice<Value = bool> + 'a,
{
    /// Values that may or may not be valid
    pub values: T::WriteSlice<'a>,

    /// Truth that each element of `values` is valid
    pub is_valid: Validity,
}
//
impl<'a, T: ArrayElement, Validity: Slice<Value = bool> + 'a> Clone
    for OptionSlice<'a, T, Validity>
{
    fn clone(&self) -> Self {
        *self
    }
}
//
impl<'a, T: ArrayElement, Validity: Slice<Value = bool> + 'a> Copy
    for OptionSlice<'a, T, Validity>
{
}
//
impl<'a, T: ArrayElement, Validity: Slice<Value = bool> + 'a> Slice
    for OptionSlice<'a, T, Validity>
{
    type Value = Option<T::WriteValue<'a>>;

    fn has_consistent_lens(&self) -> bool {
        self.values.has_consistent_lens() && self.values.len() == self.is_valid.len()
    }

    #[inline]
    fn len(&self) -> usize {
        debug_assert!(self.has_consistent_lens());
        self.is_valid.len()
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, index: usize) -> Self::Value {
        unsafe {
            self.is_valid
                .get_cloned_unchecked(index)
                .then_some(self.values.get_cloned_unchecked(index))
        }
    }

    fn iter_cloned(&self) -> impl Iterator<Item = Self::Value> + '_ {
        debug_assert!(self.has_consistent_lens());
        self.values
            .iter_cloned()
            .zip(self.is_valid.iter_cloned())
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
