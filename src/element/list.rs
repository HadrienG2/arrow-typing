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
    primitive::OptimizedValiditySlice,
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
            && self.items.len() == self.lists.total_items()
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
            let (offset, len_validity) = self.lists.get_sublist_unchecked(index);
            let (_before, start) = self.items.split_at(offset);
            let (list, _after) = start.split_at(len_validity.len());
            len_validity.apply_validity(list)
        }
    }

    fn iter_cloned(&self) -> impl Iterator<Item = Self::Element> + Clone + Debug + '_ {
        debug_assert!(self.is_consistent());
        let mut remaining = self.items;
        self.lists.iter_sublists_len_validity().map(move |sublist| {
            let (current, next) = remaining.split_at(sublist.len());
            remaining = next;
            sublist.apply_validity(current)
        })
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        debug_assert!(self.is_consistent());
        let (mid_offset, _len_validity) = self.lists.sublist_at(mid);
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
    <<Lists as SublistSlice>::LenValidity as SublistLenValidity>::ApplyValidity<Items>;

/// Slice type that can describe the layout of sub-lists within a [`ListSlice`]
#[doc(hidden)]
pub trait SublistSlice: Slice {
    /// Type of sublist
    ///
    /// This will be [`ValidSublist`] for slices of `List` and [`OptionSublist`]
    /// for slices of `Option<List>`.
    type LenValidity: SublistLenValidity;

    /// Get the total number of items across all inner sublists
    fn total_items(&self) -> usize;

    /// Get the offset, length and validity of the N-th sublist, without bounds
    /// checking
    ///
    /// # Safety
    ///
    /// Caller must ensure that `self.is_consistent()` and `index < self.len()`.
    unsafe fn get_sublist_unchecked(&self, index: usize) -> (usize, Self::LenValidity);

    /// Get the offset, length and validity of the N-th sublist, with
    /// panic-based bounds checking
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    fn sublist_at(&self, index: usize) -> (usize, Self::LenValidity) {
        assert!(self.is_consistent() && index < self.len());
        unsafe { self.get_sublist_unchecked(index) }
    }

    /// Iterate over the sublists
    fn iter_sublists_len_validity(
        &self,
    ) -> impl Iterator<Item = Self::LenValidity> + Clone + Debug + '_;
}

/// Length and validity of a sublist within [`ListSlice::items`]
#[doc(hidden)]
pub trait SublistLenValidity: Value + Eq + Hash + Ord {
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
    /// - If the `Self` type is always valid, returns `value` as-is
    /// - If the `Self` type is optionally valid and this particular `self`
    ///   instance is valid, returns `Some(value)`
    /// - If the `Self` type is optionally valid and this particular `self`
    ///   instance is invalid, returns `None`
    ///
    /// Implementations of this method should be marked `#[inline]`.
    fn apply_validity<T: Value>(&self, value: T) -> Self::ApplyValidity<T>;
}

// usize is used as a LenValidity for sublists which are always valid
impl SublistLenValidity for usize {
    #[inline]
    fn len(&self) -> usize {
        *self
    }

    type ApplyValidity<T: Value> = T;
    #[inline]
    fn apply_validity<T: Value>(&self, value: T) -> T {
        value
    }
}

/// Length and validity of a sublist which may be invalid
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[doc(hidden)]
pub struct LenValidity {
    len: usize,
    is_valid: bool,
}
//
impl LenValidity {
    /// Construct from an an optional length, assuming invalid lists are empty
    fn from_option_len(len: Option<usize>) -> Self {
        Self {
            len: len.unwrap_or(0),
            is_valid: len.is_some(),
        }
    }
}
//
impl SublistLenValidity for LenValidity {
    #[inline]
    fn len(&self) -> usize {
        self.len
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
    type LenValidity = usize;

    fn total_items(&self) -> usize {
        self.iter().sum()
    }

    unsafe fn get_sublist_unchecked(&self, index: usize) -> (usize, Self::LenValidity) {
        unsafe {
            let previous_lens = self.get_unchecked(..index);
            let offset = previous_lens.iter().sum::<usize>();
            let len = *self.get_unchecked(index);
            (offset, len)
        }
    }

    #[inline]
    fn iter_sublists_len_validity(&self) -> impl Iterator<Item = usize> + Clone + Debug + '_ {
        self.iter().cloned()
    }
}

/// Slice format used to write lists of optional lists into an Arrow builder
pub type OptionListWriteSlice<'a, Item> =
    ListSlice<<Item as ArrayElement>::WriteSlice<'a>, &'a [Option<usize>]>;
//
impl SublistSlice for &[Option<usize>] {
    type LenValidity = LenValidity;

    fn total_items(&self) -> usize {
        self.iter().fold(0, |acc, len| acc + len.unwrap_or(0))
    }

    unsafe fn get_sublist_unchecked(&self, index: usize) -> (usize, LenValidity) {
        debug_assert!(index < self.len());
        unsafe {
            let previous_lens = self.get_unchecked(..index);
            let offset = previous_lens
                .iter()
                .fold(0, |acc, len| acc + len.unwrap_or(0));
            let len = *self.get_unchecked(index);
            (offset, LenValidity::from_option_len(len))
        }
    }

    #[inline]
    fn iter_sublists_len_validity(&self) -> impl Iterator<Item = LenValidity> + Clone + Debug + '_ {
        self.iter().map(|len| LenValidity::from_option_len(*len))
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
    total_items: usize,
}
//
impl<'a, OffsetSize: OffsetSizeTrait> OffsetSublists<'a, OffsetSize> {
    /// Build from an arrow offset slice and a total number of items
    pub(crate) fn new(offsets: &'a [OffsetSize], total_items: usize) -> Self {
        Self {
            original_offsets: offsets,
            total_items,
        }
    }

    /// Corrective factor to be applied to each offset in original_offset
    #[inline]
    fn offset_shift(&self) -> usize {
        self.original_offsets
            .first()
            .map_or(0, |offset| offset.as_usize())
    }
}
//
#[doc(hidden)]
impl<OffsetSize: OffsetSizeTrait> Slice for OffsetSublists<'_, OffsetSize> {
    type Element = OffsetLen;

    #[inline]
    fn is_consistent(&self) -> bool {
        // If there are no lists, there should be no items.
        let Some((first_offset, other_offsets)) = self.original_offsets.split_first() else {
            return self.total_items == 0;
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
        (*last_offset - *first_offset).as_usize() <= self.total_items
    }

    #[inline]
    fn len(&self) -> usize {
        debug_assert!(self.is_consistent());
        self.original_offsets.len()
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, index: usize) -> Self::Element {
        debug_assert!(self.is_consistent() && index < self.len());
        unsafe {
            let offset_shift = self.original_offsets.get_unchecked(0).as_usize();
            let original_offset = self.original_offsets.get_unchecked(index).as_usize();
            let relative_offset = original_offset - offset_shift;
            let len = self.original_offsets.get(index + 1).map_or_else(
                || self.total_items - relative_offset,
                |next_original_offset| next_original_offset.as_usize() - original_offset,
            );
            OffsetLen {
                offset: relative_offset,
                len,
            }
        }
    }

    fn iter_cloned(&self) -> impl Iterator<Item = OffsetLen> + Clone + Debug + '_ {
        debug_assert!(self.is_consistent());
        let offset_shift = self.offset_shift();
        (self.original_offsets.windows(2))
            .map(move |win| OffsetLen {
                offset: win[0].as_usize() - offset_shift,
                len: (win[1] - win[0]).as_usize(),
            })
            .chain(self.original_offsets.last().map(move |last_offset| {
                let offset = last_offset.as_usize() - offset_shift;
                OffsetLen {
                    offset,
                    len: self.total_items - offset,
                }
            }))
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        debug_assert!(self.is_consistent());
        let (left_offsets, right_offsets) = self.original_offsets.split_at(mid);
        let (left_len, right_len) =
            right_offsets
                .first()
                .map_or((self.total_items, 0), |right_offset| {
                    let relative_offset = right_offset.as_usize() - self.offset_shift();
                    (relative_offset, self.total_items - relative_offset)
                });
        (
            Self {
                original_offsets: left_offsets,
                total_items: left_len,
            },
            Self {
                original_offsets: right_offsets,
                total_items: right_len,
            },
        )
    }
}
//
impl<OffsetSize: OffsetSizeTrait> SublistSlice for OffsetSublists<'_, OffsetSize> {
    type LenValidity = usize;

    #[inline]
    fn total_items(&self) -> usize {
        debug_assert!(self.is_consistent());
        self.total_items
    }

    #[inline]
    unsafe fn get_sublist_unchecked(&self, index: usize) -> (usize, usize) {
        let offset_len = self.get_cloned_unchecked(index);
        (offset_len.offset, offset_len.len)
    }

    #[inline]
    fn iter_sublists_len_validity(&self) -> impl Iterator<Item = usize> + Clone + Debug + '_ {
        self.iter_cloned().map(|offset_len| offset_len.len)
    }
}

/// Offset and length of a list within `ListSlice::items`
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[doc(hidden)]
pub struct OffsetLen {
    /// Index of the first item of the list within `ListSlice::items`
    pub offset: usize,

    /// Number of items within the list
    pub len: usize,
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
    OptionSlice<OffsetSublists<'a, OffsetSize>, OptimizedValiditySlice<Bitmap<'a>>>;
//
impl<OffsetSize: OffsetSizeTrait> SublistSlice for OptionOffsetSublists<'_, OffsetSize> {
    type LenValidity = LenValidity;

    #[inline]
    fn total_items(&self) -> usize {
        self.values.total_items()
    }

    #[inline]
    unsafe fn get_sublist_unchecked(&self, index: usize) -> (usize, LenValidity) {
        unsafe {
            let (offset, len) = self.values.get_sublist_unchecked(index);
            let is_valid = self.is_valid.get_cloned_unchecked(index);
            (offset, LenValidity { len, is_valid })
        }
    }

    #[inline]
    fn iter_sublists_len_validity(&self) -> impl Iterator<Item = LenValidity> + Clone + Debug + '_ {
        self.values
            .iter_sublists_len_validity()
            .zip(self.is_valid.iter_cloned())
            .map(|(len, is_valid)| LenValidity { len, is_valid })
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
