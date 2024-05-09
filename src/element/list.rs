//! Grouping array elements in sublists

use crate::element::{ArrayElement, Slice};
use arrow_array::{builder::GenericListBuilder, OffsetSizeTrait};
use arrow_schema::ArrowError;
use std::{fmt::Debug, marker::PhantomData};

/// A list of elements of type `Item`
///
/// Uses 32-bit signed offsets by default, which limits the sum of sublist
/// lengths to `2^31`. Use [`LargeList`] to go over this limit.
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct List<Item: ArrayElement + ?Sized, OffsetSize: OffsetSizeTrait = i32>(
    PhantomData<(Item::Value<'static>, OffsetSize)>,
);

/// A [`List`] with 64-bit offsets
pub type LargeList<Item> = List<Item, i64>;

/// Columnar alternative to a slice of slices
///
/// - The default configuration, with `usize` lengths, emulates `&[&[Item]]`: each
///   entry of `lengths` represents a sublist of a certain length within
///   `values`.
/// - The alternate [`OptionListSlice`] configuration, with `Option<usize>`,
///   emulates `&[Option<&[Item]>]` by additionally enabling null sublists to be
///   specified using `None` entries in `lengths`.
#[derive(Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ListSlice<'a, Item: ArrayElement, Length: ListLength = usize> {
    /// Concatenated elements from all inner lists
    pub values: Item::Slice<'a>,

    /// Length of each sublist within `values`, or `None` for null lists
    pub lengths: &'a [Length],
}
//
impl<'a, Item: ArrayElement, Length: ListLength> Clone for ListSlice<'a, Item, Length> {
    fn clone(&self) -> Self {
        *self
    }
}
//
impl<'a, Item: ArrayElement, Length: ListLength> Copy for ListSlice<'a, Item, Length> {}
//
impl<'a, Item: ArrayElement, Length: ListLength> Slice for ListSlice<'a, Item, Length> {
    type Value = Length::WrappedLikeSelf<Item::Slice<'a>>;

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

/// Columnar alternative to `&[Option<&[Item]>]`
pub type OptionListSlice<'a, Item> = ListSlice<'a, Item, Option<usize>>;

/// Type which can be used as a list length, i.e. `usize` or `Option<usize>`
#[doc(hidden)]
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
unsafe impl<Item: ArrayElement, OffsetSize: OffsetSizeTrait> ArrayElement
    for List<Item, OffsetSize>
{
    type BuilderBackend = GenericListBuilder<OffsetSize, Item::BuilderBackend>;
    type Value<'a> = Item::Slice<'a>;
    type Slice<'a> = ListSlice<'a, Item>;
    type ExtendFromSliceResult = Result<(), ArrowError>;
}
//
// SAFETY: Option is not a primitive type and is therefore not affected by the
//         safety precondition of ArrayElement
unsafe impl<Item: ArrayElement, OffsetSize: OffsetSizeTrait> ArrayElement
    for Option<List<Item, OffsetSize>>
{
    type BuilderBackend = GenericListBuilder<OffsetSize, Item::BuilderBackend>;
    type Value<'a> = Option<Item::Slice<'a>>;
    type Slice<'a> = OptionListSlice<'a, Item>;
    type ExtendFromSliceResult = Result<(), ArrowError>;
}

/// A [`List`] of `Item`s or an [`Option`] thereof
//
// TODO: Once supported, narrow down the bound to ArrayElement<BuilderBackend:
//       TypedBackend<Self, ExtraConfig = ListConfig<Self::Item>,
//       AlternateConfig = NoAlternateConfig> + Items and simplify the bounds of
//       TypedBuilder and BuilderConfig.
pub trait ListLike: ArrayElement {
    /// List item type
    type Item: ArrayElement;
}
//
impl<Item: ArrayElement, OffsetSize: OffsetSizeTrait> ListLike for List<Item, OffsetSize> {
    type Item = Item;
}
//
impl<Item: ArrayElement, OffsetSize: OffsetSizeTrait> ListLike for Option<List<Item, OffsetSize>> {
    type Item = Item;
}

// FIXME: Add support for fixed-size lists, whether the size is known at
//        compile-time (ConstSizedList<T, N, OffsetSize>) or at runtime
//        (FixedSizeList<T, OffsetSize>)
