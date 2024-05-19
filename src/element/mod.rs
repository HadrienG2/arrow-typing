//! Array elements and slices thereof

pub mod list;
pub mod option;
pub mod primitive;

#[cfg(doc)]
use self::option::OptionSlice;
use crate::builder::backend::TypedBackend;
#[cfg(doc)]
use crate::{bitmap::Bitmap, builder::TypedBuilder, element::primitive::PrimitiveType};
#[cfg(doc)]
use arrow_schema::ArrowError;
use std::fmt::Debug;

/// Arrow array element
///
/// # Safety
///
/// When this trait is implemented on a type that also implements the
/// [`PrimitiveType`] trait, its `WriteSlice` and `ReadSlice` associated types
/// must both be set to `&[Self]`.
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

/// A value that can be read from or written into an Arrow array
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
    fn iter_cloned(&self) -> impl Iterator<Item = Self::Element> + Clone + Debug + '_;

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

    fn iter_cloned(&self) -> impl Iterator<Item = T> + Clone + Debug + '_ {
        <[T]>::iter(self).cloned()
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        <[T]>::split_at(self, mid)
    }
}
