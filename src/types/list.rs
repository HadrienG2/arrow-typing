//! Rust mapping of Arrow's list types

use crate::{ArrayElement, OptionSlice};
use arrow_array::{builder::GenericListBuilder, OffsetSizeTrait};
use arrow_schema::ArrowError;
use std::marker::PhantomData;

/// Marker type representing an Arrow list whose elements are of type T
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct List<T: ArrayElement + ?Sized, OffsetSize: OffsetSizeTrait = i32>(
    PhantomData<(&'static T, OffsetSize)>,
);

/// A `List` whose offset buffer uses 64-bit integers
pub type LargeList<T> = List<T, i64>;

/// Columnar alternative to `&[&[T]]`
#[derive(Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ListSlice<'a, T: ArrayElement> {
    /// Concatenated elements from all inner lists
    pub values: T::Slice<'a>,

    /// Length of each sublist within `values`
    pub lengths: &'a [usize],
}
//
impl<'a, T: ArrayElement> Clone for ListSlice<'a, T> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            lengths: self.lengths,
        }
    }
}
//
impl<'a, T: ArrayElement> Copy for ListSlice<'a, T> {}

// SAFETY: List is not a primitive type and is therefore not affected by the
//         safety precondition of ArrayElement
unsafe impl<T: ArrayElement, OffsetSize: OffsetSizeTrait> ArrayElement for List<T, OffsetSize> {
    type BuilderBackend = GenericListBuilder<OffsetSize, T::BuilderBackend>;
    type Value<'a> = T::Slice<'a>;
    type Slice<'a> = ListSlice<'a, T>;
    type ExtendFromSliceResult = Result<(), ArrowError>;
}
//
// SAFETY: Option is not a primitive type and is therefore not
//         affected by the safety precondition of ArrayElement
unsafe impl<T: ArrayElement, OffsetSize: OffsetSizeTrait> ArrayElement
    for Option<List<T, OffsetSize>>
{
    type BuilderBackend = GenericListBuilder<OffsetSize, T::BuilderBackend>;
    type Value<'a> = Option<T::Slice<'a>>;
    type Slice<'a> = OptionSlice<'a, List<T, OffsetSize>>;
    type ExtendFromSliceResult = Result<(), ArrowError>;
}

// TODO: Add support for fixed-size lists, whether the size is known at
//       compile-time (ConstSizedList<T, N, OffsetSize>) or at runtime
//       (FixedSizeList<T, OffsetSize>)
