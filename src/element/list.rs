//! Grouping array elements in sublists

use crate::{
    element::{ArrayElement, Slice},
    validity::ValiditySlice,
};
use arrow_array::{builder::GenericListBuilder, OffsetSizeTrait};
use arrow_schema::ArrowError;
use std::{fmt::Debug, marker::PhantomData};

/// A list of elements of type `Item`
///
/// Uses 32-bit signed offsets by default, which limits the sum of sublist
/// lengths to `2^31`. Use [`LargeList`] to go over this limit.
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct List<Item: ArrayElement + ?Sized, OffsetSize: OffsetSizeTrait = i32>(
    PhantomData<(Item::WriteValue<'static>, OffsetSize)>,
);
//
/// A [`List`] with 64-bit offsets
pub type LargeList<Item> = List<Item, i64>;

/// Columnar alternative to a slice of slices (think `&[&[T]]`)
///
/// `values` is the concatenated list of all inner items, and `lists` specifies
/// how `values` is split into sublists. As a compromise between ergonomics and
/// efficiency, several types can be used there.
///
/// - With `&[usize]`, each entry represents the length of a sublist within
///   `values`, and sublists cannot be null. This format is used when
///   bulk-writing into a `TypedBuilder<List<_>>`.
/// - With `&[Option<usize>]`, each entry represents either the length of a
///   valid sublist, or `None` to denote a null sublist. This format is used
///   when bulk-writing into a `TypedBuilder<Option<List<_>>>`.
/// - [`OffsetLengths`] is an opaque type that behaves like `&[usize]`, but uses
///   a different storage layout that is suitable for in-place readout from
///   Arrow arrays. This format is used when bulk-reading from a
///   `TypedArray<List<_>>`.
/// - [`OptionOffsetLengths`] is another opaque type that behaves like
///   `&[Option<usize>]` but uses a different internal storage format that is
///   suitable for in-place readout from Arrow arrays. This format is used when
///   bulk-reading from a `TypedArray<Option<List<_>>>`.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ListSlice<Items: Slice, Lists: Sublists> {
    /// Concatenated items from all inner lists
    pub items: Items,

    /// Layout of sub-lists within `items`
    pub lists: Lists,
}
//
impl<Items: Slice, Lists: Sublists> ListSlice<Items, Lists> {
    crate::inherent_slice_methods!(has_consistent_lens, element: ListSliceElement<Items, Lists>);
}
//
/// Sublist type returned by the slice API of ListSlice
///
/// Will be an `Items` sublist for slices of `List` and `Option<Items>` for
/// slices of `Option<List>`.
pub type ListSliceElement<Items, Lists> =
    <<Lists as Sublists>::Length as ListLength>::WrappedLikeSelf<Items>;
//
impl<Items: Slice, Lists: Sublists> Slice for ListSlice<Items, Lists> {
    type Value = ListSliceElement<Items, Lists>;

    fn has_consistent_lens(&self) -> bool {
        self.items.has_consistent_lens()
            && self.lists.has_consistent_lens()
            && self.items.len()
                == self
                    .lists
                    .iter_lengths()
                    .map(|len| len.as_len())
                    .sum::<usize>()
    }

    #[inline]
    fn len(&self) -> usize {
        debug_assert!(self.has_consistent_lens());
        self.lists.len()
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, index: usize) -> Self::Value {
        unsafe { self.iter_cloned().nth(index).unwrap_unchecked() }
    }

    fn iter_cloned(&self) -> impl Iterator<Item = Self::Value> + '_ {
        debug_assert!(self.has_consistent_lens());
        let mut remaining = self.items;
        self.lists.iter_lengths().map(move |len| {
            let (current, next) = remaining.split_at(len.as_len());
            remaining = next;
            len.wrap_like_self(current)
        })
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        debug_assert!(self.has_consistent_lens());
        let (left_lists, right_lists) = self.lists.split_at(mid);
        let left_items_len = left_lists
            .iter_lengths()
            .map(|len| len.as_len())
            .sum::<usize>();
        let (left_items, right_items) = self.items.split_at(left_items_len);
        (
            Self {
                items: left_items,
                lists: left_lists,
            },
            Self {
                items: right_items,
                lists: right_lists,
            },
        )
    }
}

/// Slice type that can model a list of sublists
///
/// This trait is destined to be replaced with a `Slice<Value: ListLength>`
/// bound once rustc supports them.
//
// TODO: Replace members with Slice<Value: ListLength> bound when supported
#[doc(hidden)]
pub trait Sublists: Slice {
    /// Length of a sublist
    type Length: ListLength;

    /// Iterate over the lengths of inner sublists
    fn iter_lengths(&self) -> impl Iterator<Item = Self::Length>;
}

/// Type which can be used to count a list length and possibly specify a null
/// list, i.e. `usize` or `Option<usize>`
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

/// Slice format used to write lists of lists into an Arrow builder
pub type ListWriteSlice<'a, Item> = ListSlice<<Item as ArrayElement>::WriteSlice<'a>, &'a [usize]>;
//
/// Slice format used to write lists of optional lists into an Arrow builder
pub type OptionListWriteSlice<'a, Item> =
    ListSlice<<Item as ArrayElement>::WriteSlice<'a>, &'a [Option<usize>]>;
//
impl Sublists for &[usize] {
    type Length = usize;
    fn iter_lengths(&self) -> impl Iterator<Item = usize> {
        self.iter_cloned()
    }
}
//
impl Sublists for &[Option<usize>] {
    type Length = Option<usize>;
    fn iter_lengths(&self) -> impl Iterator<Item = Option<usize>> {
        self.iter_cloned()
    }
}

/// Slice format used to access lists of lists from Arrow storage
pub type ListReadSlice<'a, Item, OffsetSize = i32> =
    ListSlice<<Item as ArrayElement>::ReadSlice<'a>, OffsetLengths<'a, OffsetSize>>;
//
/// Like `ListReadSlice`, but inner sublists use 64-bit offsets
pub type LargeListReadSlice<'a, Item> = ListReadSlice<'a, Item, i64>;
//
/// Sublist lengths in native Arrow storage format
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct OffsetLengths<'a, OffsetSize: OffsetSizeTrait> {
    /// Start of each sublist
    offsets: &'a [OffsetSize],

    /// Total number of elements across all sublists
    total_len: usize,
}
//
impl<OffsetSize: OffsetSizeTrait> Slice for OffsetLengths<'_, OffsetSize> {
    type Value = usize;

    fn has_consistent_lens(&self) -> bool {
        if let Some(last) = self.offsets.last() {
            last.as_usize() <= self.total_len
        } else {
            self.total_len == 0
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.offsets.len()
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, index: usize) -> usize {
        unsafe {
            let next_start = if index == self.offsets.len() - 1 {
                self.total_len
            } else {
                self.offsets.get_unchecked(index + 1).as_usize()
            };
            next_start
                .checked_sub(self.offsets.get_unchecked(index).as_usize())
                .expect("list offsets should be sorted in increasing order")
        }
    }

    #[inline]
    fn iter_cloned(&self) -> impl Iterator<Item = Self::Value> + '_ {
        (self.offsets.windows(2))
            .map(|win| win[1].as_usize() - win[0].as_usize())
            .chain(
                self.offsets
                    .last()
                    .map(|last_offset| self.total_len - last_offset.as_usize()),
            )
    }

    fn split_at(&self, index: usize) -> (Self, Self) {
        let (left_starts, right_starts) = self.offsets.split_at(index);
        let (left_len, right_len) = if let Some(right_offset) = right_starts.first() {
            (
                right_offset.as_usize(),
                self.total_len - right_offset.as_usize(),
            )
        } else {
            (self.total_len, 0)
        };
        (
            Self {
                offsets: left_starts,
                total_len: left_len,
            },
            Self {
                offsets: right_starts,
                total_len: right_len,
            },
        )
    }
}
//
impl<OffsetSize: OffsetSizeTrait> Sublists for OffsetLengths<'_, OffsetSize> {
    type Length = usize;
    fn iter_lengths(&self) -> impl Iterator<Item = usize> {
        self.iter_cloned()
    }
}

/// Slice format used to access lists of optional lists from Arrow storage
pub type OptionListReadSlice<'a, Item, OffsetSize = i32> =
    ListSlice<<Item as ArrayElement>::ReadSlice<'a>, OptionOffsetLengths<'a, OffsetSize>>;
//OffsetSize: OffsetSizeTrait
/// Like `OptionListReadSlice`, but inner sublists use 64-bit offsets
pub type OptionLargeListReadSlice<'a, Item> = OptionListReadSlice<'a, Item, i64>;
//
/// Slice of lists in native arrow format (null buffer + offset buffer)
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct OptionOffsetLengths<'a, OffsetSize: OffsetSizeTrait> {
    /// Null buffer
    validity: ValiditySlice<'a>,

    /// Offset-based sublist lengths
    offset_lengths: OffsetLengths<'a, OffsetSize>,
}
//
impl<OffsetSize: OffsetSizeTrait> Slice for OptionOffsetLengths<'_, OffsetSize> {
    type Value = Option<usize>;

    fn has_consistent_lens(&self) -> bool {
        self.offset_lengths.has_consistent_lens()
            && self.validity.len() == self.offset_lengths.len()
    }

    #[inline]
    fn len(&self) -> usize {
        self.offset_lengths.len()
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, index: usize) -> Option<usize> {
        unsafe {
            self.validity
                .get_cloned_unchecked(index)
                .then_some(self.offset_lengths.get_cloned_unchecked(index))
        }
    }

    fn iter_cloned(&self) -> impl Iterator<Item = Option<usize>> + '_ {
        (self.validity.iter())
            .zip(self.offset_lengths.iter_lengths())
            .map(|(valid, length)| valid.then_some(length))
    }

    fn split_at(&self, index: usize) -> (Self, Self) {
        let (left_validity, right_validity) = self.validity.split_at(index);
        let (left_offsets, right_offsets) = self.offset_lengths.split_at(index);
        (
            Self {
                validity: left_validity,
                offset_lengths: left_offsets,
            },
            Self {
                validity: right_validity,
                offset_lengths: right_offsets,
            },
        )
    }
}
//
impl<OffsetSize: OffsetSizeTrait> Sublists for OptionOffsetLengths<'_, OffsetSize> {
    type Length = Option<usize>;
    fn iter_lengths(&self) -> impl Iterator<Item = Self::Length> {
        self.iter_cloned()
    }
}

// SAFETY: List is not a primitive type and is therefore not affected by the
//         safety precondition of ArrayElement
unsafe impl<Item: ArrayElement, OffsetSize: OffsetSizeTrait> ArrayElement
    for List<Item, OffsetSize>
{
    type BuilderBackend = GenericListBuilder<OffsetSize, Item::BuilderBackend>;
    type WriteValue<'a> = Item::WriteSlice<'a>;
    type ReadValue<'a> = Item::ReadSlice<'a>;
    type WriteSlice<'a> = ListWriteSlice<'a, Item>;
    type ReadSlice<'a> = ListReadSlice<'a, Item, OffsetSize>;
    type ExtendFromSliceResult = Result<(), ArrowError>;
}
//
// SAFETY: Option is not a primitive type and is therefore not affected by the
//         safety precondition of ArrayElement
unsafe impl<Item: ArrayElement, OffsetSize: OffsetSizeTrait> ArrayElement
    for Option<List<Item, OffsetSize>>
{
    type BuilderBackend = GenericListBuilder<OffsetSize, Item::BuilderBackend>;
    type WriteValue<'a> = Option<Item::WriteSlice<'a>>;
    type ReadValue<'a> = Option<Item::ReadSlice<'a>>;
    type WriteSlice<'a> = OptionListWriteSlice<'a, Item>;
    type ReadSlice<'a> = OptionListReadSlice<'a, Item, OffsetSize>;
    type ExtendFromSliceResult = Result<(), ArrowError>;
}

/// A [`List`] of `Item`s or an [`Option`] thereof
//
// TODO: Once supported, narrow down the bound to ArrayElement<BuilderBackend:
//       TypedBackend<Self, ExtraConfig = ListConfig<Self::Item>,
//       AlternateConfig = NoAlternateConfig> + Items and simplify the bounds of
//       TypedBuilder and BuilderConfig accordingly.
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
