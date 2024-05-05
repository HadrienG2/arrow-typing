//! Rust mapping of Arrow's list types

use crate::{ArrayElement, Slice};
use arrow_array::{builder::GenericListBuilder, OffsetSizeTrait};
use arrow_schema::ArrowError;
use std::{fmt::Debug, marker::PhantomData};

/// Marker type representing an Arrow list whose elements are of type T
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct List<T: ArrayElement + ?Sized, OffsetSize: OffsetSizeTrait = i32>(
    PhantomData<(T::Value<'static>, OffsetSize)>,
);

/// A [`List`] with a 64-bit element count
pub type LargeList<T> = List<T, i64>;

/// Columnar alternative to `&[&[T]]` (by default) or `&[Option<&[T]>]` (in the
/// [`OptionListSlice`] variant).
#[derive(Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ListSlice<'a, T: ArrayElement, Length: ListLength = usize> {
    /// Concatenated elements from all inner lists
    pub values: T::Slice<'a>,

    /// Length of each sublist within `values`
    pub lengths: &'a [Length],
}
//
impl<'a, T: ArrayElement, Length: ListLength> Clone for ListSlice<'a, T, Length> {
    fn clone(&self) -> Self {
        *self
    }
}
//
impl<'a, T: ArrayElement, Length: ListLength> Copy for ListSlice<'a, T, Length> {}
//
impl<'a, T: ArrayElement, Length: ListLength> Slice for ListSlice<'a, T, Length> {
    type Value = Length::WrappedLikeSelf<T::Slice<'a>>;

    fn has_consistent_lens(&self) -> bool {
        self.values.has_consistent_lens()
            && self.values.len() == self.lengths.iter().map(ListLength::as_len).sum::<usize>()
    }

    fn len(&self) -> usize {
        debug_assert!(self.has_consistent_lens());
        self.lengths.len()
    }

    fn iter_cloned(&self) -> impl Iterator<Item = Self::Value> + '_ {
        debug_assert!(self.has_consistent_lens());
        let mut remaining = self.values;
        self.lengths.iter_cloned().map(move |len| {
            let (current, next) = remaining.split_at(len.as_len());
            remaining = next;
            len.wrap_like_self(current)
        })
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        debug_assert!(self.has_consistent_lens());
        let (left_lengths, right_lengths) = self.lengths.split_at(mid);
        let left_values_len = left_lengths.iter().map(ListLength::as_len).sum::<usize>();
        let (left_values, right_values) = self.values.split_at(left_values_len);
        (
            Self {
                values: left_values,
                lengths: left_lengths,
            },
            Self {
                values: right_values,
                lengths: right_lengths,
            },
        )
    }
}

/// Columnar alternative to `&[Option<&[T]>]`
///
/// Each entry of `lengths` that is `None` creates a null sublist.
pub type OptionListSlice<'a, T> = ListSlice<'a, T, Option<usize>>;

/// Type which can be used as a list length
pub trait ListLength: Copy + Clone + Debug {
    /// Return type of `wrap_like_self`, see that for more info
    type WrappedLikeSelf<T: Debug>: Debug;

    /// Wrap the input value in the same layers of optionality as `self`
    ///
    /// - If `Self` is not an `Option`, returns `value` as-is
    /// - If `self` is `Some(len)`, returns `Some(value)`
    /// - If `self` is `None`, returns `None`
    fn wrap_like_self<T: Debug>(&self, value: T) -> Self::WrappedLikeSelf<T>;

    /// Interpret `self` as a sublist length, treating null sublists (sublists
    /// of optional length `None`) as sublists of length 0.
    fn as_len(&self) -> usize;
}
//
impl ListLength for usize {
    type WrappedLikeSelf<T: Debug> = T;

    #[inline]
    fn wrap_like_self<T: Debug>(&self, value: T) -> T {
        value
    }

    #[inline]
    fn as_len(&self) -> usize {
        *self
    }
}
//
impl ListLength for Option<usize> {
    type WrappedLikeSelf<T: Debug> = Option<T>;

    fn wrap_like_self<T: Debug>(&self, value: T) -> Option<T> {
        self.map(|_| value)
    }

    #[inline]
    fn as_len(&self) -> usize {
        self.unwrap_or(0)
    }
}

// SAFETY: List is not a primitive type and is therefore not affected by the
//         safety precondition of ArrayElement
unsafe impl<T: ArrayElement, OffsetSize: OffsetSizeTrait> ArrayElement for List<T, OffsetSize> {
    type BuilderBackend = GenericListBuilder<OffsetSize, T::BuilderBackend>;
    type Value<'a> = T::Slice<'a>;
    type Slice<'a> = ListSlice<'a, T>;
    type ExtendFromSliceResult = Result<(), ArrowError>;
}
//
// SAFETY: Option is not a primitive type and is therefore not affected by the
//         safety precondition of ArrayElement
unsafe impl<T: ArrayElement, OffsetSize: OffsetSizeTrait> ArrayElement
    for Option<List<T, OffsetSize>>
{
    type BuilderBackend = GenericListBuilder<OffsetSize, T::BuilderBackend>;
    type Value<'a> = Option<T::Slice<'a>>;
    type Slice<'a> = OptionListSlice<'a, T>;
    type ExtendFromSliceResult = Result<(), ArrowError>;
}

// TODO: Add support for fixed-size lists, whether the size is known at
//       compile-time (ConstSizedList<T, N, OffsetSize>) or at runtime
//       (FixedSizeList<T, OffsetSize>)
