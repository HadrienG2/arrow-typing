//! A layer on top of [`arrow`](https://docs.rs/arrow) which enables arrow
//! arrays to be built and accessed using strongly typed Rust APIs.

pub mod builder;
pub mod types;
pub mod validity;

#[cfg(doc)]
use crate::types::primitive::PrimitiveType;
use crate::types::primitive::{
    Date32, Date64, Duration, IntervalDayTime, IntervalMonthDayNano, IntervalYearMonth,
    Microsecond, Millisecond, Nanosecond, Null, Second, Time,
};
use arrow_array::builder::{
    BooleanBuilder, Date32Builder, Date64Builder, DurationMicrosecondBuilder,
    DurationMillisecondBuilder, DurationNanosecondBuilder, DurationSecondBuilder, Float16Builder,
    Float32Builder, Float64Builder, Int16Builder, Int32Builder, Int64Builder, Int8Builder,
    IntervalDayTimeBuilder, IntervalMonthDayNanoBuilder, IntervalYearMonthBuilder,
    Time32MillisecondBuilder, Time32SecondBuilder, Time64MicrosecondBuilder,
    Time64NanosecondBuilder, UInt16Builder, UInt32Builder, UInt64Builder, UInt8Builder,
};
use arrow_schema::ArrowError;
use half::f16;
use std::fmt::Debug;

pub use builder::TypedBuilder;

/// Strongly typed data which can be stored as an Arrow array element
pub trait ArrayElement: Debug + Send + Sync + 'static {
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
//
impl NullableElement for Null {}
//
impl<T: ArrayElement> NullableElement for Option<T> where Option<T>: ArrayElement {}

/// [`ArrayElement`] which can be read or written in bulk using slices
///
/// # Safety
///
/// If this trait is implemented on a [primitive type](PrimitiveType), then the
/// `Slice` associated type **must** be set to `&[Self]`.
//
// FIXME: The bound I actually want is `ArrayElement<BuilderBackend:
//        ExtendFromSlice<Self>>`, use that once associated type bounds are
//        stable (stabilization PR has landed on nightly at time of writing),
//        and then remove all the unnecessary ExtendFromSlice bounds in builder
//        backends that leak into user-visible APIs.
pub unsafe trait SliceElement: ArrayElement {
    /// Slice type used for bulk insertion and readout
    ///
    /// For simple types this will just be `&[Self::Value]`, but for more
    /// complex types, efficiency constraints may dictate a different layout.
    ///
    /// For example, nullable primitive types like `Option<u16>` are
    /// bulk-manipulated using [`OptionSlice`] batches. And tuple types like
    /// `(T, U, V)` are bulk-manipulated using `(&[T], &[U], &[V])` batches.
    type Slice<'a>;

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

/// Columnar alternative to `&[Option<T>]`
#[derive(Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OptionSlice<'a, T: SliceElement> {
    /// Values that may or may not be valid
    pub values: T::Slice<'a>,

    /// Truth that each element of `values` is valid
    pub is_valid: &'a [bool],
}
//
impl<'a, T: SliceElement> Clone for OptionSlice<'a, T>
where
    T::Slice<'a>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            is_valid: self.is_valid,
        }
    }
}

// Enable strongly typed arrays of primitive types
macro_rules! unsafe_impl_primitive_element {
    ($($element:ty => $builder:ty),*) => {
        $(
            impl ArrayElement for $element {
                type BuilderBackend = $builder;
                type Value<'a> = Self;
            }

            // SAFETY: By construction, it is enforced that Slice is &[Self]
            unsafe impl SliceElement for $element {
                type Slice<'a> = &'a [Self];
                type ExtendFromSliceResult = ();
            }

            // FIXME: I tried to make this blanket-impl'd for Option<T> where
            //        T::BuilderBackend: TypedBackend<Option<T>>, but this
            //        caused problems down the line where backends were not
            //        recognized by the trait solver as implementing
            //        TypedBackend<Option<T>> because Option<T> did not
            //        implement ArrayElement. Let's keep this macrofied for now.
            impl ArrayElement for Option<$element> {
                type BuilderBackend = $builder;
                type Value<'a> = Option<$element>;
            }

            // SAFETY: Option is not a primitive type and is therefore not
            //         affected by the safety precondition of SliceElement
            unsafe impl SliceElement for Option<$element> {
                type Slice<'a> = OptionSlice<'a, $element>;
                type ExtendFromSliceResult = Result<(), ArrowError>;
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

/// Shared test utilities
#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    /// Maximum array length/capacity used in unit tests
    pub const MAX_CAPACITY: usize = 256;

    /// Generate a capacity between 0 and MAX_CAPACITY
    pub fn length_or_capacity() -> impl Strategy<Value = usize> {
        0..=MAX_CAPACITY
    }
}
