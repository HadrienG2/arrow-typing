//! A layer on top of [`arrow`](https://docs.rs/arrow) which enables arrow
//! arrays to be built and accessed using strongly typed Rust APIs.

pub mod builder;
pub mod types;

use crate::types::primitive::{
    Date32, Date64, Duration, IntervalDayTime, IntervalMonthDayNano, IntervalYearMonth,
    Microsecond, Millisecond, Nanosecond, Second, Time,
};
#[cfg(doc)]
use crate::{builder::TypedBuilder, types::Null};
use arrow_array::builder::{
    BooleanBuilder, Date32Builder, Date64Builder, DurationMicrosecondBuilder,
    DurationMillisecondBuilder, DurationNanosecondBuilder, DurationSecondBuilder, Float16Builder,
    Float32Builder, Float64Builder, Int16Builder, Int32Builder, Int64Builder, Int8Builder,
    IntervalDayTimeBuilder, IntervalMonthDayNanoBuilder, IntervalYearMonthBuilder,
    Time32MillisecondBuilder, Time32SecondBuilder, Time64MicrosecondBuilder,
    Time64NanosecondBuilder, UInt16Builder, UInt32Builder, UInt64Builder, UInt8Builder,
};
use arrow_schema::ArrowError;
use builder::backend::TypedBackend;
use half::f16;
use std::fmt::Debug;

/// Strongly typed data which can be stored as an arrow array element
pub trait ArrayElement: Send + Sync + 'static {
    /// Array builder implementation
    type BuilderBackend: builder::backend::TypedBackend<Self>;

    /// Array element type used for individual element writes and reads
    ///
    /// For simple types, this will just be `Self`. But for more complex types,
    /// type system and/or efficiency constraints may force us to use a
    /// different type.
    ///
    /// For example, lists of primitive types T are best read and written as
    /// slices `&[T]`.
    type Value<'a>;
}

/// [`ArrayElement`] which has a null value
///
/// This trait is implemented for both the null element type [`Null`] and
/// options of valid array element types. It enables efficient bulk insertion of
/// null values via [`TypedBuilder::extend_with_nulls()`].
pub trait NullableElement: ArrayElement {}
impl<T: ArrayElement> NullableElement for Option<T> where Option<T>: ArrayElement {}

/// [`ArrayElement`] which can be read or written in bulk using slices
///
/// # Safety
///
/// If this trait is implemented on a primitive type, then Slice<'a> **must**
/// be defined as &[Self].
//
// FIXME: The bound I actually want is `ArrayElement<BuilderBackend:
//        ExtendFromSlice<Self>>`, use that once associated type bounds are
//        stable (stabilization PR has landed on nightly at time of writing),
//        and then remove all the unnecessary ExtendFromSlice bounds in
//        user-visible APIs.
pub unsafe trait SliceElement: ArrayElement {
    /// Slice type used for bulk insertion and readout
    ///
    /// For simple types this will just be `&[Self]`, but for more complex
    /// types, efficiency constraints may dictate a different layout.
    ///
    /// For example, nullable primitive types like `Option<u16>` are
    /// bulk-manipulated using [`OptionSlice`] batches. And tuple types like
    /// `(T, U, V)` are bulk-manipulated using `(&[T], &[U], &[V])` batches.
    type Slice<'a>;

    /// Return type of [`TypedBuilder::extend_from_slice()`].
    ///
    /// Bulk insertion always succeeds for simple types. But for complex types
    /// which need composite slice types like `(&[T], &[U])`, bulk insertion can
    /// fail with `ArrowError` if the inner slices have unequal length.
    ///
    /// Accordingly, the return type of `extend_from_slice()` is `()` for
    /// simple slices, and `Result<(), ArrowError>` for composite slices.
    type ExtendFromSliceResult: Debug;
}

/// Alternative to `&[Option<T>]` that is friendlier to columnar storage
pub struct OptionSlice<'a, T: SliceElement> {
    /// Values that may or may not be valid
    pub values: T::Slice<'a>,

    /// Truth that each element of `values` is valid
    pub is_valid: &'a [bool],
}

// Enable strongly typed arrays of primitive types
macro_rules! unsafe_impl_primitive_element {
    ($($element:ty => $builder:ty),*) => {
        $(
            impl ArrayElement for $element {
                type BuilderBackend = $builder;
                type Value<'a> = Self;
            }
            unsafe impl SliceElement for $element {
                type Slice<'a> = &'a [Self];
                type ExtendFromSliceResult = ();
            }
        )*
    };
}
//
// SAFETY: The macro ensures that SliceElement::Slice is &[Self]
unsafe_impl_primitive_element!(
    bool => BooleanBuilder,
    Date32 => Date32Builder,
    Date64 => Date64Builder,
    // TODO: Support decimals, see types module for rustc blocker info.
    Duration<Microsecond> => DurationMicrosecondBuilder,
    Duration<Millisecond> => DurationMillisecondBuilder,
    Duration<Nanosecond> => DurationNanosecondBuilder,
    Duration<Second> => DurationSecondBuilder,
    f16 => Float16Builder,
    f32 => Float32Builder,
    f64 => Float64Builder,
    i8 => Int8Builder,
    i16 => Int16Builder,
    i32 => Int32Builder,
    i64 => Int64Builder,
    IntervalDayTime => IntervalDayTimeBuilder,
    IntervalMonthDayNano => IntervalMonthDayNanoBuilder,
    IntervalYearMonth => IntervalYearMonthBuilder,
    Time<Millisecond> => Time32MillisecondBuilder,
    Time<Second> => Time32SecondBuilder,
    Time<Microsecond> => Time64MicrosecondBuilder,
    Time<Nanosecond> => Time64NanosecondBuilder,
    // TODO: Support timestamps, see types module for rustc blocker info.
    u8 => UInt8Builder,
    u16 => UInt16Builder,
    u32 => UInt32Builder,
    u64 => UInt64Builder
);

// Enabled strongly typed arrays of optional types
impl<T: ArrayElement> ArrayElement for Option<T>
where
    T::BuilderBackend: TypedBackend<Option<T>>,
{
    type BuilderBackend = T::BuilderBackend;
    type Value<'a> = Option<T::Value<'a>>;
}
//
// SAFETY: Option is not a primitive type and is therefore not affected by the
//         safety precondition of SliceElement on primitive types.
unsafe impl<T: SliceElement> SliceElement for Option<T>
where
    T::BuilderBackend: TypedBackend<Option<T>>,
{
    type Slice<'a> = OptionSlice<'a, T>;
    type ExtendFromSliceResult = Result<(), ArrowError>;
}
