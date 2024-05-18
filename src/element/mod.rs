//! Array elements and slices thereof

pub mod list;
pub mod primitive;

use crate::{bitmap::Bitmap, builder::backend::TypedBackend, element::primitive::Null};
#[cfg(doc)]
use crate::{builder::TypedBuilder, element::primitive::PrimitiveType};
#[cfg(doc)]
use arrow_schema::ArrowError;
use std::fmt::Debug;

/// Arrow array element
///
/// # Safety
///
/// When this trait is implemented on a type that also implements the
/// [`PrimitiveType`] trait, its `WriteSlice` associated type must be set to
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
    type WriteValue<'a>: Value;

    /// Array element type used when reading from an array
    ///
    /// For simple types, this will be the same as `WriteValue`. But for more
    /// complex types, especially those involving lists, the two types can
    /// differ.
    ///
    /// The reason why this happens is that in-place reads from Arrow arrays
    /// must follow the underlying Arrow storage format to be efficient, even
    /// when using this format comes at the expense of reduced ergonomics and
    /// compatibility with Rust idoms. For example...
    ///
    /// - [Slices of optional items](OptionSlice) must use the bit-packed
    ///   [`Bitmap`] null buffer format, instead of the more idiomatic unpacked
    ///   `&[bool]` format.
    /// - Slices of lists must use a signed offset buffer to encode the position
    ///   and size of inner sublists, instead of a more idiomatic array of
    ///   sublist lengths.
    ///
    /// In constrast, writes are (relatively) free to use idiomatic Rust data
    /// structures that are easier for a user to construct.
    ///
    /// Ultimately, `WriteValue` and `ReadValue` will always be two
    /// implementations of the same logical concept (e.g. if `WriteValue` is a
    /// slice of booleans, `ReadValue` will be a slice of booleans as well). It
    /// just happens that the implementations of this concept may differ.
    ///
    /// Although it is currently impossible to express this at the Rust type
    /// system level, we guarantee that if one of `ReadValue` an `WriteValue`
    /// implements `PartialEq`, then they will both do so and also will be
    /// comparable with each other.
    type ReadValue<'a>: Value;

    /// Slice type used for bulk insertion into an array
    ///
    /// For simple types this will just be `&[Self::WriteValue]`. But for more
    /// complex types, efficiency constraints may dictate a different layout.
    ///
    /// For example, nullable primitive types like `Option<u16>` are
    /// bulk-manipulated using [`OptionSlice`] batches. And tuple types like
    /// `(T, U, V)` are bulk-manipulated using `(&[T], &[U], &[V])` batches.
    type WriteSlice<'a>: Slice<Element = Self::WriteValue<'a>>;

    /// Slice type used for bulk readout from an array
    ///
    /// `ReadSlice` is to `WriteSlice` what `ReadValue` is to `WriteValue`: an
    /// alternate implementation of the same logical concept which is optimized
    /// for efficient Arrow data access, rather than array-building ergonomics.
    type ReadSlice<'a>: Slice<Element = Self::ReadValue<'a>>;

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

/// A value that can be stored in an Arrow array
pub trait Value: Clone + Copy + Debug + Default + Send + Sized + Sync {}
//
impl<T: Clone + Copy + Debug + Default + Send + Sized + Sync> Value for T {}

/// A Rust slice `&[T]` or a columnar generalization thereof
///
/// Columnar slice types like [`OptionSlice`] are internally composed of
/// multiple Rust slices. Before using such a composite slice through this trait
/// or the inherent methods that mirror it, you should make sure that the inner
/// slices are consistent with each other (i.e. have compatible length) using
/// the `is_consistent()` method.
///
/// If the inner slices are not consistent, the other methods of this trait will
/// return unpredictable results. Because the results can be often right and
/// rarely wrong, it is strongly recommended that implementations of these other
/// methods start with a `debug_assert!(self.is_consistent())` debug assertion
/// in order to detect such incorrect usage in testing environments.
pub trait Slice: Value {
    /// Individual slice element
    type Element: Value;

    /// Truth that all inner slices are consistent with each other
    ///
    /// Implementations of this method should be marked `#[inline]`.
    fn is_consistent(&self) -> bool;

    /// Number of slice elements
    ///
    /// Implementations of this method should be marked `#[inline]`.
    fn len(&self) -> usize;

    /// Truth that this slice has no elements
    #[inline]
    fn is_empty(&self) -> bool {
        debug_assert!(self.is_consistent());
        self.len() == 0
    }

    /// Value of the first element of the slice, or `None` if the slice is empty
    #[inline]
    fn first_cloned(&self) -> Option<Self::Element> {
        debug_assert!(self.is_consistent());
        self.get_cloned(0)
    }

    /// Value of the last element of the slice, or `None` if it is empty
    #[inline]
    fn last_cloned(&self) -> Option<Self::Element> {
        debug_assert!(self.is_consistent());
        self.len()
            .checked_sub(1)
            .and_then(|last_idx| self.get_cloned(last_idx))
    }

    /// Value of the `index`-th slice element, if in bounds
    #[inline]
    fn get_cloned(&self, index: usize) -> Option<Self::Element> {
        debug_assert!(self.is_consistent());
        (self.is_consistent() && index < self.len())
            // SAFETY: Preconditions are checked above
            .then(|| unsafe { self.get_cloned_unchecked(index) })
    }

    /// Value of the `index`-th slice element, without bounds checking
    ///
    /// For a safe alternative see [`get_cloned`](Self::get_cloned).
    ///
    /// Implementations of this method should be marked `#[inline]`.
    ///
    /// # Safety
    ///
    /// Callers must ensure that `self.is_consistent()` is true and that `index
    /// < self.len()`.
    unsafe fn get_cloned_unchecked(&self, index: usize) -> Self::Element;

    /// Value of the `index`-th slice element, with panic-based bounds checking
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    fn at(&self, index: usize) -> Self::Element {
        debug_assert!(self.is_consistent());
        self.get_cloned(index).expect("index is out of bounds")
    }

    /// Iterate over copies of the elements of this slice
    fn iter_cloned(&self) -> impl Iterator<Item = Self::Element> + '_;

    /// Split the slice into two subslices at `mid`
    ///
    /// # Panics
    ///
    /// Panics if the slice has less than `mid` elements.
    fn split_at(&self, mid: usize) -> (Self, Self);
}
//
impl<T: Value> Slice for &[T] {
    type Element = T;

    #[inline]
    fn is_consistent(&self) -> bool {
        true
    }

    #[inline]
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, index: usize) -> T {
        unsafe { *<[T]>::get_unchecked(self, index) }
    }

    fn iter_cloned(&self) -> impl Iterator<Item = T> + '_ {
        <[T]>::iter(self).cloned()
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        <[T]>::split_at(self, mid)
    }
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

/// A columnar slice of `Option<T>`
///
/// This type is logically equivalent to `&[Option<Values::Element>]`, but it is
/// implemented as the combination of a slice of data with a slice of booleans
/// that tell whether each row of data is valid or not.
///
/// Validity can be tracked using either a standard `&[bool]` Rust slice of
/// booleans or a bit-packed [`Bitmap`].
#[derive(Clone, Copy, Debug, Default, Hash)]
pub struct OptionSlice<Values: Slice, Validity: Slice<Element = bool>> {
    /// Truth that each element of `values` is valid
    pub is_valid: Validity,

    /// Values that may or may not be valid
    pub values: Values,
}
//
impl<Values: Slice, Validity: Slice<Element = bool>> OptionSlice<Values, Validity> {
    crate::inherent_slice_methods!(is_consistent, element: Option<Values::Element>);
}
//
impl<Values: Slice, Validity: Slice<Element = bool>> Eq for OptionSlice<Values, Validity> where
    Values::Element: Eq
{
}
//
impl<Values: Slice, Validity: Slice<Element = bool>> Ord for OptionSlice<Values, Validity>
where
    Values::Element: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}
//
impl<Values: Slice, Validity: Slice<Element = bool>, OtherSlice: Slice> PartialEq<OtherSlice>
    for OptionSlice<Values, Validity>
where
    Option<Values::Element>: PartialEq<OtherSlice::Element>,
{
    fn eq(&self, other: &OtherSlice) -> bool {
        self.iter().eq(other.iter_cloned())
    }
}
//
impl<Values: Slice, Validity: Slice<Element = bool>, OtherSlice: Slice> PartialOrd<OtherSlice>
    for OptionSlice<Values, Validity>
where
    Option<Values::Element>: PartialOrd<OtherSlice::Element>,
{
    fn partial_cmp(&self, other: &OtherSlice) -> Option<std::cmp::Ordering> {
        self.iter().partial_cmp(other.iter_cloned())
    }
}
//
impl<Values: Slice, Validity: Slice<Element = bool>> Slice for OptionSlice<Values, Validity> {
    type Element = Option<Values::Element>;

    #[inline]
    fn is_consistent(&self) -> bool {
        self.values.is_consistent()
            && self.is_valid.is_consistent()
            && self.values.len() == self.is_valid.len()
    }

    #[inline]
    fn len(&self) -> usize {
        debug_assert!(self.is_consistent());
        self.is_valid.len()
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, index: usize) -> Self::Element {
        debug_assert!(self.is_consistent() && index < self.len());
        unsafe {
            self.is_valid
                .get_cloned_unchecked(index)
                .then_some(self.values.get_cloned_unchecked(index))
        }
    }

    fn iter_cloned(&self) -> impl Iterator<Item = Self::Element> + '_ {
        debug_assert!(self.is_consistent());
        self.values
            .iter_cloned()
            .zip(self.is_valid.iter_cloned())
            .map(|(value, is_valid)| is_valid.then_some(value))
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        debug_assert!(self.is_consistent());
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
/// OptionSlice layout used when writing to a `TypedBuilder<Option<T>>`
///
/// Uses a [`WriteSlice`](ArrayElement::WriteSlice) of `T` for bulk-insertion of
/// values of type `T`, and a simple `&[bool]` to tell whether each element of
/// the data slice is valid/null or not.
pub type OptionWriteSlice<'a, T> = OptionSlice<<T as ArrayElement>::WriteSlice<'a>, &'a [bool]>;
//
/// OptionSlice layout used when reading from an `Array<Option<T>>`
///
/// Follows Arrow's internal storage format to allow for in-place data access,
/// which means using a [`ReadSlice`](ArrayElement::ReadSlice) of `T` for
/// bulk-readout and a [`Bitmap`] for validity tracking.
pub type OptionReadSlice<'a, T> = OptionSlice<<T as ArrayElement>::ReadSlice<'a>, Bitmap<'a>>;
