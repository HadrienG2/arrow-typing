//! Arrays of optional values

use super::{ArrayElement, Slice};
#[cfg(doc)]
use crate::builder::TypedBuilder;
use crate::{bitmap::Bitmap, element::primitive::Null};
use std::fmt::Debug;

/// [`ArrayElement`] which has a null value
///
/// This trait is implemented for both the null element type [`Null`] and
/// options of valid array element types.
///
/// It enables efficient bulk insertion of null values via
/// [`TypedBuilder::extend_with_nulls()`].
pub trait NullableElement: ArrayElement {}
//
impl NullableElement for Null {}
//
impl<T: ArrayElement> NullableElement for Option<T> where Option<T>: ArrayElement {}

/// [`ArrayElement`] for which `Option<Self>` is an Arrow-supported storage type
///
/// This trait is implemented for almost every Arrow-supported storage type
/// except for [`Null`] and `Option<_>`, so you can think of it as the logical
/// opposite of [`NullableElement`].
///
/// It is used to expose `Option<_>`-specific array builder methods.
///
/// # Safety
///
/// `Self::ValiditySlice` must match the backend-specific validity slice type.
//
// TODO: Once the Rust type system supports it, enforce that
//       Option<T>: NullableElement<
//          BuilderBackend = BuilderBackend<T>,
//          ReadValue<'a> = Option<T::ReadValue<'a>>,
//          WriteValue<'a> = Option<T::WriteValue<'a>>,
//          ReadSlice<'a> = OptionReadSlice<'a, T>>,
//          WriteSlice<'a> = OptionWriteSlice<'a, T>>,
//          ExtendFromSliceResult = Result<(), ArrowError>
//       >
//       BuilderBackend<T>::ValiditySlice = Self::ValiditySlice
//       must hold and remove the corresponding bounds around the codebase.
pub unsafe trait OptionalElement: ArrayElement {
    /// Null buffer representation used for optional values of this type
    ///
    /// For all simple types, this will be a [`Bitmap`].
    type ValiditySlice<'a>: Slice<Element = bool> + Eq + PartialEq<&'a [bool]>;
}

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
