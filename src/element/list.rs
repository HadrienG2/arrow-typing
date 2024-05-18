//! Grouping array elements in sublists

use crate::{
    bitmap::Bitmap,
    element::{ArrayElement, Slice},
};
use arrow_array::{builder::GenericListBuilder, OffsetSizeTrait};
use arrow_schema::ArrowError;
use std::{fmt::Debug, hash::Hash, marker::PhantomData};

use super::{
    option::{OptionSlice, OptionalElement},
    Value,
};

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

/// A columnar slice of lists
///
/// `values` is the concatenation of all inner lists, and `lists` specifies how
/// `values` is split into sublists. As a compromise between ergonomics and
/// efficiency, several types of `Lists` are supported.
///
/// - If `Lists` is `&[usize]`, each entry represents the length of a sublist
///   within `values`, and sublists cannot be null. This format is used when
///   bulk-writing into a `TypedBuilder<List>`.
/// - With `&[Option<usize>]`, each entry represents either the length of a
///   valid sublist, or `None` to denote a null sublist. This format is used
///   when bulk-writing into a `TypedBuilder<Option<List>>`.
/// - [`OffsetSublists`] is an opaque type that behaves like `&[usize]`, but
///   uses a storage layout that is suitable for in-place readout from Arrow
///   arrays. This format is used when bulk-reading from a `TypedArray<List>`.
/// - [`OptionOffsetSublists`] is another opaque type that behaves like
///   `&[Option<usize>]` but uses a different internal storage format that is
///   suitable for in-place readout from Arrow arrays. This format is used when
///   bulk-reading from a `TypedArray<Option<List>>`.
#[derive(Clone, Copy, Debug, Default, Hash)]
pub struct ListSlice<Items: Slice, Lists: SublistSlice> {
    /// Concatenated items from all inner lists
    pub items: Items,

    /// Layout of sub-lists within `items`
    pub lists: Lists,
}
//
impl<Items: Slice, Lists: SublistSlice> ListSlice<Items, Lists> {
    crate::inherent_slice_methods!(is_consistent, element: ListSliceElement<Items, Lists>);
}
//
impl<Items: Slice, Lists: SublistSlice> Eq for ListSlice<Items, Lists> where
    ListSliceElement<Items, Lists>: Eq
{
}
//
impl<Items: Slice, Lists: SublistSlice> Ord for ListSlice<Items, Lists>
where
    ListSliceElement<Items, Lists>: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}
//
impl<Items: Slice, Lists: SublistSlice, OtherSlice: Slice> PartialEq<OtherSlice>
    for ListSlice<Items, Lists>
where
    ListSliceElement<Items, Lists>: PartialEq<OtherSlice::Element>,
{
    fn eq(&self, other: &OtherSlice) -> bool {
        self.iter().eq(other.iter_cloned())
    }
}
//
impl<Items: Slice, Lists: SublistSlice, OtherSlice: Slice> PartialOrd<OtherSlice>
    for ListSlice<Items, Lists>
where
    ListSliceElement<Items, Lists>: PartialOrd<OtherSlice::Element>,
{
    fn partial_cmp(&self, other: &OtherSlice) -> Option<std::cmp::Ordering> {
        self.iter().partial_cmp(other.iter_cloned())
    }
}
//
impl<Items: Slice, Lists: SublistSlice> Slice for ListSlice<Items, Lists> {
    type Element = ListSliceElement<Items, Lists>;

    #[inline]
    fn is_consistent(&self) -> bool {
        self.items.is_consistent()
            && self.lists.is_consistent()
            && self.items.len()
                == self
                    .lists
                    .last_sublist()
                    .map_or(0, |sublist| sublist.offset() + sublist.len())
    }

    #[inline]
    fn len(&self) -> usize {
        debug_assert!(self.is_consistent());
        self.lists.len()
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, index: usize) -> Self::Element {
        debug_assert!(self.is_consistent() && index < self.len());
        unsafe {
            let sublist = self.lists.get_sublist_unchecked(index);
            let (_before, start) = self.items.split_at(sublist.offset());
            let (list, _after) = start.split_at(sublist.len());
            sublist.apply_validity(list)
        }
    }

    fn iter_cloned(&self) -> impl Iterator<Item = Self::Element> + '_ {
        debug_assert!(self.is_consistent());
        let mut remaining = self.items;
        self.lists.iter_sublists().map(move |sublist| {
            let (current, next) = remaining.split_at(sublist.len());
            remaining = next;
            sublist.apply_validity(current)
        })
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        debug_assert!(self.is_consistent());
        let mid_offset = self.lists.sublist_at(mid).offset();
        let (left_lists, right_lists) = self.lists.split_at(mid);
        let (left_items, right_items) = self.items.split_at(mid_offset);
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
//
/// Sublist type returned by the slice API of [`ListSlice`]
///
/// Will be a sub-slice of `items` for slices of [`List`] and an `Option<Items>`
/// optional sub-slice of items for slices of `Option<List>`.
pub type ListSliceElement<Items, Lists> =
    <<Lists as SublistSlice>::Sublist as Sublist>::ApplyValidity<Items>;

/// Slice type that can describe the layout of sub-lists within a [`ListSlice`]
#[doc(hidden)]
pub trait SublistSlice: Slice {
    /// Type of sublist
    ///
    /// This will be [`ValidSublist`] for slices of `List` and [`OptionSublist`]
    /// for slices of `Option<List>`.
    type Sublist: Sublist;

    /// Get the last sublist in this slice, if any
    fn last_sublist(&self) -> Option<Self::Sublist>;

    /// Get the N-th sublist, without bounds checking
    ///
    /// Implementations of this method should be marked `#[inline]`.
    ///
    /// # Safety
    ///
    /// Caller must ensure that `self.is_consistent()` and `index < self.len()`.
    unsafe fn get_sublist_unchecked(&self, index: usize) -> Self::Sublist;

    /// Get the N-th sublist, with panic-based bounds checking
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    fn sublist_at(&self, index: usize) -> Self::Sublist {
        assert!(self.is_consistent() && index < self.len());
        unsafe { self.get_sublist_unchecked(index) }
    }

    /// Iterate over the sublists
    fn iter_sublists(&self) -> impl Iterator<Item = Self::Sublist>;
}

/// Sublist within [`ListSlice::items`]
#[doc(hidden)]
pub trait Sublist: Value + Eq + Hash + PartialEq + PartialOrd {
    /// Position of the first item within `ListSlice::items`
    ///
    /// Implementations of this method should be marked `#[inline]`.
    fn offset(&self) -> usize;

    /// Length of the sublist in items
    ///
    /// Implementations of this method should be marked `#[inline]`.
    fn len(&self) -> usize;

    /// Truth that this sublist is empty
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return type of `apply_validity`, see that function for more info
    type ApplyValidity<T: Value>: Value;

    /// Wrap `value` in the same layers of optionality as `self`
    ///
    /// - If `Self` is always valid, returns `value` as-is
    /// - If `Self` is optionally valid and `self` is valid, returns
    ///   `Some(value)`
    /// - If `Self` is optionally valid and `self` is invalid, returns `None`
    ///
    /// Implementations of this method should be marked `#[inline]`.
    fn apply_validity<T: Value>(&self, value: T) -> Self::ApplyValidity<T>;
}

/// Sublist which is always valid
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[doc(hidden)]
pub struct ValidSublist {
    offset: usize,
    len: usize,
}
//
impl Sublist for ValidSublist {
    #[inline]
    fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    type ApplyValidity<T: Value> = T;
    #[inline]
    fn apply_validity<T: Value>(&self, value: T) -> T {
        value
    }
}

/// Sublist which may or may not be valid
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[doc(hidden)]
pub struct OptionSublist {
    sublist: ValidSublist,
    is_valid: bool,
}
//
impl OptionSublist {
    /// Construct from an offset and an optional length
    fn from_option_len(offset: usize, len: Option<usize>) -> Self {
        Self {
            sublist: ValidSublist {
                offset,
                len: len.unwrap_or(0),
            },
            is_valid: len.is_some(),
        }
    }
}
//
impl Sublist for OptionSublist {
    #[inline]
    fn offset(&self) -> usize {
        self.sublist.offset
    }

    #[inline]
    fn len(&self) -> usize {
        self.sublist.len
    }

    type ApplyValidity<T: Value> = Option<T>;
    #[inline]
    fn apply_validity<T: Value>(&self, value: T) -> Option<T> {
        self.is_valid.then_some(value)
    }
}

/// Slice format used to write lists of lists into an Arrow builder
pub type ListWriteSlice<'a, Item> = ListSlice<<Item as ArrayElement>::WriteSlice<'a>, &'a [usize]>;
//
impl SublistSlice for &[usize] {
    type Sublist = ValidSublist;

    fn last_sublist(&self) -> Option<ValidSublist> {
        let (&len, previous_lens) = self.split_last()?;
        let offset = previous_lens.iter().sum::<usize>();
        Some(ValidSublist { offset, len })
    }

    #[inline]
    unsafe fn get_sublist_unchecked(&self, index: usize) -> ValidSublist {
        unsafe {
            let previous_lens = self.get_unchecked(..index);
            let offset = previous_lens.iter().sum::<usize>();
            let len = *self.get_unchecked(index);
            ValidSublist { offset, len }
        }
    }

    fn iter_sublists(&self) -> impl Iterator<Item = ValidSublist> {
        let mut offset = 0;
        self.iter().map(move |&len| {
            let result = ValidSublist { offset, len };
            offset += len;
            result
        })
    }
}

/// Slice format used to write lists of optional lists into an Arrow builder
pub type OptionListWriteSlice<'a, Item> =
    ListSlice<<Item as ArrayElement>::WriteSlice<'a>, &'a [Option<usize>]>;
//
impl SublistSlice for &[Option<usize>] {
    type Sublist = OptionSublist;

    fn last_sublist(&self) -> Option<Self::Sublist> {
        let (&len, previous_lens) = self.split_last()?;
        let offset = previous_lens
            .iter()
            .fold(0, |acc, len| acc + len.unwrap_or(0));
        Some(OptionSublist::from_option_len(offset, len))
    }

    #[inline]
    unsafe fn get_sublist_unchecked(&self, index: usize) -> OptionSublist {
        debug_assert!(index < self.len());
        unsafe {
            let previous_lens = self.get_unchecked(..index);
            let offset = previous_lens
                .iter()
                .fold(0, |acc, len| acc + len.unwrap_or(0));
            let len = *self.get_unchecked(index);
            OptionSublist::from_option_len(offset, len)
        }
    }

    fn iter_sublists(&self) -> impl Iterator<Item = OptionSublist> {
        let mut offset = 0;
        self.iter().map(move |&len| {
            let result = OptionSublist::from_option_len(offset, len);
            offset += len.unwrap_or(0);
            result
        })
    }
}

/// Slice format used to access lists of lists from Arrow storage
pub type ListReadSlice<'a, Item, OffsetSize = i32> =
    ListSlice<<Item as ArrayElement>::ReadSlice<'a>, OffsetSublists<'a, OffsetSize>>;
//
/// Like [`ListReadSlice`], but inner sublists use 64-bit offsets
pub type LargeListReadSlice<'a, Item> = ListReadSlice<'a, Item, i64>;
//
/// Slice of non-optional sublists in the native offset-based Arrow format
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct OffsetSublists<'a, OffsetSize: OffsetSizeTrait> {
    /// Start of each sublist within the **original** [`ListSlice`], before any
    /// splitting occurred
    ///
    /// Arrow's offset-based sublist representation has the major benefit of
    /// enabling O(1) random access to any given sublist, but this comes at the
    /// cost of making slice splitting less intuitive.
    ///
    /// More specifically, the i-th offset within the current
    /// [`ListSlice::items`] slice is `original_offsets[i] -
    /// original_offsets[0]`, and not `original_offsets[i]` as you might expect.
    original_offsets: &'a [OffsetSize],

    /// Total number of items
    total_len: usize,
}
//
impl<OffsetSize: OffsetSizeTrait> OffsetSublists<'_, OffsetSize> {
    /// Corrective factor to be applied to each offset in original_offset
    fn offset_shift(&self) -> usize {
        self.original_offsets
            .first()
            .map_or(0, |offset| offset.as_usize())
    }
}
//
#[doc(hidden)]
impl<OffsetSize: OffsetSizeTrait> Slice for OffsetSublists<'_, OffsetSize> {
    type Element = ValidSublist;

    #[inline]
    fn is_consistent(&self) -> bool {
        // If there are no lists, there should be no items.
        let Some((first_offset, other_offsets)) = self.original_offsets.split_first() else {
            return self.total_len == 0;
        };

        // Offsets should be sorted in increasing order
        let mut last_offset = first_offset;
        for offset in other_offsets {
            if offset < last_offset {
                return false;
            }
            last_offset = offset;
        }

        // Offsets should not overflow the items list
        (*last_offset - *first_offset).as_usize() <= self.total_len
    }

    #[inline]
    fn len(&self) -> usize {
        debug_assert!(self.is_consistent());
        self.original_offsets.len()
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, index: usize) -> ValidSublist {
        debug_assert!(self.is_consistent() && index < self.len());
        unsafe {
            let offset_shift = self.original_offsets.get_unchecked(0).as_usize();
            let original_offset = self.original_offsets.get_unchecked(index).as_usize();
            let relative_offset = original_offset - offset_shift;
            let next_idx = index + 1;
            let len = if next_idx < self.original_offsets.len() {
                self.original_offsets.get_unchecked(next_idx).as_usize() - original_offset
            } else {
                self.total_len - relative_offset
            };
            ValidSublist {
                offset: relative_offset,
                len,
            }
        }
    }

    fn iter_cloned(&self) -> impl Iterator<Item = ValidSublist> + '_ {
        debug_assert!(self.is_consistent());
        let offset_shift = self.offset_shift();
        (self.original_offsets.windows(2))
            .map(move |win| ValidSublist {
                offset: win[0].as_usize() - offset_shift,
                len: (win[1] - win[0]).as_usize(),
            })
            .chain(self.original_offsets.last().map(move |last_offset| {
                let offset = last_offset.as_usize() - offset_shift;
                ValidSublist {
                    offset,
                    len: self.total_len - offset,
                }
            }))
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        debug_assert!(self.is_consistent());
        let (left_offsets, right_offsets) = self.original_offsets.split_at(mid);
        let (left_len, right_len) =
            right_offsets
                .first()
                .map_or((self.total_len, 0), |right_offset| {
                    let relative_offset = right_offset.as_usize() - self.offset_shift();
                    (relative_offset, self.total_len - relative_offset)
                });
        (
            Self {
                original_offsets: left_offsets,
                total_len: left_len,
            },
            Self {
                original_offsets: right_offsets,
                total_len: right_len,
            },
        )
    }
}
//
impl<OffsetSize: OffsetSizeTrait> SublistSlice for OffsetSublists<'_, OffsetSize> {
    type Sublist = ValidSublist;

    #[inline]
    fn last_sublist(&self) -> Option<ValidSublist> {
        self.last_cloned()
    }

    #[inline]
    unsafe fn get_sublist_unchecked(&self, index: usize) -> ValidSublist {
        self.get_cloned_unchecked(index)
    }

    #[inline]
    fn iter_sublists(&self) -> impl Iterator<Item = ValidSublist> {
        self.iter_cloned()
    }
}

/// Slice format used to access lists of optional lists from Arrow storage
pub type OptionListReadSlice<'a, Item, OffsetSize = i32> =
    ListSlice<<Item as ArrayElement>::ReadSlice<'a>, OptionOffsetSublists<'a, OffsetSize>>;
//
/// Like [`OptionListReadSlice`], but inner sublists use 64-bit offsets
pub type OptionLargeListReadSlice<'a, Item> = OptionListReadSlice<'a, Item, i64>;
//
/// Slice of optional sublists in the native offset-based Arrow format
pub type OptionOffsetSublists<'a, OffsetSize> =
    OptionSlice<OffsetSublists<'a, OffsetSize>, Bitmap<'a>>;
//
impl<OffsetSize: OffsetSizeTrait> SublistSlice for OptionOffsetSublists<'_, OffsetSize> {
    type Sublist = OptionSublist;

    fn last_sublist(&self) -> Option<Self::Sublist> {
        debug_assert!(self.is_consistent());
        let sublist = self.values.last_sublist()?;
        let is_valid = self.is_valid.last()?;
        Some(OptionSublist { sublist, is_valid })
    }

    #[inline]
    unsafe fn get_sublist_unchecked(&self, index: usize) -> OptionSublist {
        debug_assert!(self.is_consistent() && index < self.len());
        unsafe {
            let sublist = self.values.get_sublist_unchecked(index);
            let is_valid = self.is_valid.get_cloned_unchecked(index);
            OptionSublist { sublist, is_valid }
        }
    }

    fn iter_sublists(&self) -> impl Iterator<Item = OptionSublist> {
        debug_assert!(self.is_consistent());
        self.values
            .iter_sublists()
            .zip(self.is_valid.iter_cloned())
            .map(|(sublist, is_valid)| OptionSublist { sublist, is_valid })
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
// SAFETY: GenericListBuilder does use a Bitmap validity slice
unsafe impl<Item: ArrayElement, OffsetSize: OffsetSizeTrait> OptionalElement
    for List<Item, OffsetSize>
{
    type ValiditySlice<'a> = Bitmap<'a>;
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
