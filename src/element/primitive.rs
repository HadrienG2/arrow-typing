//! Primitive array element types

use super::{option::OptionalElement, ArrayElement, Slice, Value};
use crate::{bitmap::Bitmap, impl_option_element};
use arrow_array::{
    builder::{BooleanBuilder, NullBuilder, PrimitiveBuilder},
    types::*,
};
use half::f16;
#[cfg(any(test, feature = "proptest"))]
use proptest::prelude::*;
use std::{cmp::Ordering, fmt::Debug, hash::Hash, marker::PhantomData, num::TryFromIntError};

// === Arrow primitive types other than DataTypes ===

/// A value that is always null
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Null;
//
// SAFETY: Null is not a PrimitiveType and is therefore not concerned by
//         ArrayElement's safety contract.
unsafe impl ArrayElement for Null {
    type BuilderBackend = NullBuilder;
    type ReadValue<'a> = Self;
    type WriteValue<'a> = Self;
    type ReadSlice<'a> = UniformSlice<Null>;
    type WriteSlice<'a> = UniformSlice<Null>;
    type ExtendFromSliceResult = ();
}

// SAFETY: bool is not a PrimitiveType and is therefore not concerned by
//         ArrayElement's safety contract.
unsafe impl ArrayElement for bool {
    type BuilderBackend = BooleanBuilder;
    type WriteValue<'a> = Self;
    type ReadValue<'a> = Self;
    type WriteSlice<'a> = &'a [Self];
    type ReadSlice<'a> = &'a [Self];
    type ExtendFromSliceResult = ();
}
//
// SAFETY: BooleanBuilder does use a Bitmap validity slice
unsafe impl OptionalElement for bool {
    type ValiditySlice<'a> = Bitmap<'a>;
}
//
impl_option_element!(bool);

/// A pseudo-slice whose elements are all equal to the same run-time constant
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord)]
pub struct UniformSlice<T: Value> {
    /// Common value of each element of the slice
    element: T,

    /// Number of elements "contained" by the slice
    len: usize,
}
//
impl<T: Value> UniformSlice<T> {
    /// Set up a uniform slice with `len` elements, all equal to `element`
    pub fn new(element: T, len: usize) -> Self {
        Self { element, len }
    }

    crate::inherent_slice_methods!(element: T);
}
//
impl<T: Value, S: Slice> PartialEq<S> for UniformSlice<T>
where
    T: PartialEq<S::Element>,
{
    fn eq(&self, other: &S) -> bool {
        self.iter().eq(other.iter_cloned())
    }
}
//
impl<T: Value, S: Slice> PartialOrd<S> for UniformSlice<T>
where
    T: PartialOrd<S::Element>,
{
    fn partial_cmp(&self, other: &S) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter_cloned())
    }
}
//
impl<T: Value> Slice for UniformSlice<T> {
    type Element = T;

    #[inline]
    fn is_consistent(&self) -> bool {
        true
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, _index: usize) -> T {
        self.element
    }

    fn iter_cloned(&self) -> impl Iterator<Item = Self::Element> + '_ {
        std::iter::repeat(self.element).take(self.len)
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.len, "split point is above total element count");
        (
            Self {
                element: self.element,
                len: mid,
            },
            Self {
                element: self.element,
                len: self.len - mid,
            },
        )
    }
}

/// A pseudo-slice whose elements are all equal to the same compile-time boolean
//
// TODO: Once Rust gets some version of the adt_const_params nightly feature,
//       generalize to a `ConstSlice<T, const ELEMENT: T>(usize)` and use that
//       for the ReadSlice and WriteSlice of Null.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord)]
pub struct ConstBoolSlice<const ELEMENT: bool>(usize);
//
impl<const ELEMENT: bool> ConstBoolSlice<ELEMENT> {
    /// Set up a uniform slice with `len` elements, all equal to `ELEMENT`
    pub fn new(len: usize) -> Self {
        Self(len)
    }

    crate::inherent_slice_methods!(element: bool);
}
//
impl<const ELEMENT: bool, S: Slice> PartialEq<S> for ConstBoolSlice<ELEMENT>
where
    S::Element: PartialEq<bool>,
{
    fn eq(&self, other: &S) -> bool {
        other.iter_cloned().eq(self.iter())
    }
}
//
impl<const ELEMENT: bool, S: Slice> PartialOrd<S> for ConstBoolSlice<ELEMENT>
where
    S::Element: PartialOrd<bool>,
{
    fn partial_cmp(&self, other: &S) -> Option<Ordering> {
        other.iter_cloned().partial_cmp(self.iter())
    }
}
//
impl<const ELEMENT: bool> Slice for ConstBoolSlice<ELEMENT> {
    type Element = bool;

    #[inline]
    fn is_consistent(&self) -> bool {
        true
    }

    #[inline]
    fn len(&self) -> usize {
        self.0
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, _index: usize) -> bool {
        ELEMENT
    }

    fn iter_cloned(&self) -> impl Iterator<Item = bool> + '_ {
        std::iter::repeat(ELEMENT).take(self.0)
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.0, "split point is above total element count");
        (Self(mid), Self(self.0 - mid))
    }
}

// === Strong value types matching non-std Arrow DataTypes ===

/// Date type representing the elapsed time since the UNIX epoch in days
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Date<T>(T);
//
#[cfg(any(test, feature = "proptest"))]
impl<T: Arbitrary> Arbitrary for Date<T> {
    type Parameters = T::Parameters;
    type Strategy = prop::strategy::Map<T::Strategy, fn(T) -> Self>;
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        T::arbitrary_with(args).prop_map(Self)
    }
}
//
// One direction of conversion is easy...
impl<T> From<T> for Date<T> {
    #[inline(always)]
    fn from(value: T) -> Self {
        Self(value)
    }
}
//
// ...but the other cannot be done generically without violating the orphan rule
pub type Date32 = Date<i32>;
pub type Date64 = Date<i64>;
//
impl From<Date32> for i32 {
    #[inline(always)]
    fn from(value: Date32) -> Self {
        value.0
    }
}
//
impl From<Date64> for i64 {
    #[inline(always)]
    fn from(value: Date64) -> Self {
        value.0
    }
}

// TODO: Waiting for adt_const_params rustc feature to be able to expose the
//       desired strongly typed version of the Decimal type family:
//
//       #[derive(Clone, Copy, Debug)]
//       #[repr(transparent)]
//       pub struct Decimal<
//           T: DecimalRepr,
//           const PRECISION: Option<u8> = None,
//           const SCALE: Option<i8> = None,
//       >(T);

/// Measure of elapsed time with a certain integer unit
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Duration<Unit: TimeUnit>(i64, PhantomData<Unit>);
//
#[cfg(any(test, feature = "proptest"))]
impl<Unit: TimeUnit> Arbitrary for Duration<Unit> {
    type Parameters = <i64 as Arbitrary>::Parameters;
    type Strategy = prop::strategy::Map<<i64 as Arbitrary>::Strategy, fn(i64) -> Self>;
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        i64::arbitrary_with(args).prop_map(|inner| Self(inner, PhantomData))
    }
}
//
impl<Unit: TimeUnit> From<i64> for Duration<Unit> {
    #[inline(always)]
    fn from(value: i64) -> Self {
        Self(value, PhantomData)
    }
}
//
impl<Unit: TimeUnit> From<Duration<Unit>> for i64 {
    #[inline(always)]
    fn from(value: Duration<Unit>) -> Self {
        value.0
    }
}
//
type StdDuration = std::time::Duration;
//
impl TryFrom<Duration<Second>> for StdDuration {
    type Error = TryFromIntError;
    #[inline]
    fn try_from(value: Duration<Second>) -> Result<Self, Self::Error> {
        u64::try_from(value.0).map(StdDuration::from_secs)
    }
}
//
impl TryFrom<StdDuration> for Duration<Second> {
    type Error = TryFromIntError;
    #[inline]
    fn try_from(value: StdDuration) -> Result<Self, Self::Error> {
        i64::try_from(value.as_secs()).map(|secs| Self(secs, PhantomData))
    }
}
//
impl TryFrom<Duration<Millisecond>> for StdDuration {
    type Error = TryFromIntError;
    #[inline]
    fn try_from(value: Duration<Millisecond>) -> Result<Self, Self::Error> {
        u64::try_from(value.0).map(StdDuration::from_millis)
    }
}
//
impl TryFrom<StdDuration> for Duration<Millisecond> {
    type Error = TryFromIntError;
    #[inline]
    fn try_from(value: StdDuration) -> Result<Self, Self::Error> {
        i64::try_from(value.as_millis()).map(|millis| Self(millis, PhantomData))
    }
}
//
impl TryFrom<Duration<Microsecond>> for StdDuration {
    type Error = TryFromIntError;
    #[inline]
    fn try_from(value: Duration<Microsecond>) -> Result<Self, Self::Error> {
        u64::try_from(value.0).map(StdDuration::from_micros)
    }
}
//
impl TryFrom<StdDuration> for Duration<Microsecond> {
    type Error = TryFromIntError;
    #[inline]
    fn try_from(value: StdDuration) -> Result<Self, Self::Error> {
        i64::try_from(value.as_micros()).map(|micros| Self(micros, PhantomData))
    }
}
//
impl TryFrom<Duration<Nanosecond>> for StdDuration {
    type Error = TryFromIntError;
    #[inline]
    fn try_from(value: Duration<Nanosecond>) -> Result<Self, Self::Error> {
        u64::try_from(value.0).map(StdDuration::from_nanos)
    }
}
//
impl TryFrom<StdDuration> for Duration<Nanosecond> {
    type Error = TryFromIntError;
    #[inline]
    fn try_from(value: StdDuration) -> Result<Self, Self::Error> {
        i64::try_from(value.as_nanos()).map(|nanos| Self(nanos, PhantomData))
    }
}

/// "Calendar" time interval in days and milliseconds
#[derive(Clone, Copy, Debug, Default)]
#[repr(transparent)]
pub struct IntervalDayTime(i64);
//
impl IntervalDayTime {
    /// Creates a IntervalDayTime
    ///
    /// # Arguments
    ///
    /// * `days` - The number of days (+/-) represented in this interval
    /// * `millis` - The number of milliseconds (+/-) represented in this interval
    #[inline]
    pub fn new(days: i32, millis: i32) -> Self {
        Self(IntervalDayTimeType::make_value(days, millis))
    }

    /// Turns a IntervalDayTime into a tuple of (days, milliseconds)
    #[inline]
    pub fn to_parts(self) -> (i32, i32) {
        IntervalDayTimeType::to_parts(self.0)
    }
}
//
#[cfg(any(test, feature = "proptest"))]
impl Arbitrary for IntervalDayTime {
    type Parameters = <i64 as Arbitrary>::Parameters;
    type Strategy = prop::strategy::Map<<i64 as Arbitrary>::Strategy, fn(i64) -> Self>;
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        i64::arbitrary_with(args).prop_map(Self)
    }
}
//
impl From<i64> for IntervalDayTime {
    #[inline(always)]
    fn from(value: i64) -> Self {
        Self(value)
    }
}
//
impl From<IntervalDayTime> for i64 {
    #[inline(always)]
    fn from(value: IntervalDayTime) -> Self {
        value.0
    }
}

/// "Calendar" time interval in months, days and nanoseconds
#[derive(Clone, Copy, Debug, Default)]
#[repr(transparent)]
pub struct IntervalMonthDayNano(i128);
//
impl IntervalMonthDayNano {
    /// Creates a IntervalMonthDayNano
    ///
    /// # Arguments
    ///
    /// * `months` - The number of months (+/-) represented in this interval
    /// * `days` - The number of days (+/-) represented in this interval
    /// * `nanos` - The number of nanoseconds (+/-) represented in this interval
    #[inline]
    pub fn new(months: i32, days: i32, nanos: i64) -> Self {
        Self(IntervalMonthDayNanoType::make_value(months, days, nanos))
    }

    /// Turns a IntervalMonthDayNano into a tuple of (months, days, nanoseconds)
    #[inline]
    pub fn to_parts(self) -> (i32, i32, i64) {
        IntervalMonthDayNanoType::to_parts(self.0)
    }
}
//
#[cfg(any(test, feature = "proptest"))]
impl Arbitrary for IntervalMonthDayNano {
    type Parameters = <i128 as Arbitrary>::Parameters;
    type Strategy = prop::strategy::Map<<i128 as Arbitrary>::Strategy, fn(i128) -> Self>;
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        i128::arbitrary_with(args).prop_map(Self)
    }
}
//
impl From<i128> for IntervalMonthDayNano {
    #[inline(always)]
    fn from(value: i128) -> Self {
        Self(value)
    }
}
//
impl From<IntervalMonthDayNano> for i128 {
    #[inline(always)]
    fn from(value: IntervalMonthDayNano) -> Self {
        value.0
    }
}

/// "Calendar" time interval stored as a number of whole months
#[derive(Clone, Copy, Debug, Default)]
#[repr(transparent)]
pub struct IntervalYearMonth(i32);
//
impl IntervalYearMonth {
    /// Creates a IntervalYearMonth
    ///
    /// # Arguments
    ///
    /// * `years` - The number of years (+/-) represented in this interval
    /// * `months` - The number of months (+/-) represented in this interval
    #[inline]
    pub fn new(years: i32, months: i32) -> Self {
        Self(IntervalYearMonthType::make_value(years, months))
    }

    /// Turns a IntervalYearMonth into a number of months
    ///
    /// This operation is technically a no-op, it is included for
    /// comprehensiveness.
    #[inline]
    pub fn to_months(self) -> i32 {
        self.0
    }
}
//
#[cfg(any(test, feature = "proptest"))]
impl Arbitrary for IntervalYearMonth {
    type Parameters = <i32 as Arbitrary>::Parameters;
    type Strategy = prop::strategy::Map<<i32 as Arbitrary>::Strategy, fn(i32) -> Self>;
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        i32::arbitrary_with(args).prop_map(Self)
    }
}
//
impl From<i32> for IntervalYearMonth {
    #[inline(always)]
    fn from(value: i32) -> Self {
        Self(value)
    }
}
//
impl From<IntervalYearMonth> for i32 {
    #[inline(always)]
    fn from(value: IntervalYearMonth) -> Self {
        value.0
    }
}

/// Elapsed time since midnight
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Time<Unit: TimeUnit>(<Unit as TimeUnit>::TimeStorage);
//
#[cfg(any(test, feature = "proptest"))]
impl<Unit: TimeUnit> Arbitrary for Time<Unit>
where
    Unit::TimeStorage: Arbitrary,
{
    type Parameters = <Unit::TimeStorage as Arbitrary>::Parameters;

    type Strategy = prop::strategy::Map<
        <Unit::TimeStorage as Arbitrary>::Strategy,
        fn(Unit::TimeStorage) -> Self,
    >;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        Unit::TimeStorage::arbitrary_with(args).prop_map(Self)
    }
}
//
impl From<i32> for Time<Second> {
    #[inline(always)]
    fn from(value: i32) -> Self {
        Self(value)
    }
}
//
impl From<i32> for Time<Millisecond> {
    #[inline(always)]
    fn from(value: i32) -> Self {
        Self(value)
    }
}
//
impl From<i64> for Time<Microsecond> {
    #[inline(always)]
    fn from(value: i64) -> Self {
        Self(value)
    }
}
//
impl From<i64> for Time<Nanosecond> {
    #[inline(always)]
    fn from(value: i64) -> Self {
        Self(value)
    }
}
//
impl From<Time<Second>> for i32 {
    #[inline(always)]
    fn from(value: Time<Second>) -> Self {
        value.0
    }
}
//
impl From<Time<Millisecond>> for i32 {
    #[inline(always)]
    fn from(value: Time<Millisecond>) -> Self {
        value.0
    }
}
//
impl From<Time<Microsecond>> for i64 {
    #[inline(always)]
    fn from(value: Time<Microsecond>) -> Self {
        value.0
    }
}
//
impl From<Time<Nanosecond>> for i64 {
    #[inline(always)]
    fn from(value: Time<Nanosecond>) -> Self {
        value.0
    }
}

// TODO: Waiting for adt_const_params rustc feature and a constified Arc
//       constructor to be able to expose the desired strongly typed version of
//       the Timestamp type family:
//
//       #[derive(Clone, Copy, Debug)]
//       #[repr(transparent)]
//       pub struct Timestamp<
//           Unit: TimeUnit,
//           const TIMESTAMP: Option<Arc<str>> = None,
//       >(i64);

/// Unit of time
pub trait TimeUnit: Debug {
    /// Storage format for time since midnight in this unit
    type TimeStorage: Value + Eq + Hash + Ord;
}

/// Second duration storage granularity
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Second;
//
impl TimeUnit for Second {
    type TimeStorage = i32;
}

/// Millisecond duration storage granularity
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Millisecond;
//
impl TimeUnit for Millisecond {
    type TimeStorage = i32;
}

/// Microsecond duration storage granularity
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Microsecond;
//
impl TimeUnit for Microsecond {
    type TimeStorage = i64;
}

/// Nanosecond duration storage granularity
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Nanosecond;
//
impl TimeUnit for Nanosecond {
    type TimeStorage = i64;
}

// === Equivalent of ArrowPrimitiveType for the types defined in this module ===

/// Strong value type with a corresponding [`ArrowPrimitiveType`]
///
/// # Safety
///
/// The type for which this trait is implemented must be a `repr(transparent)`
/// wrapper over the underlying `ArrowPrimitiveType::NativeType`.
pub unsafe trait PrimitiveType:
    // TODO: Once Rust's trait solver supports it, use an
    //       ArrayElement<XyzValue<'_> = Self, XyzSlice<'_> = &[Self]> bound to
    //       simplify downstream usage and remove the unsafe contract of
    //       ArrayElement.
    ArrayElement<BuilderBackend = PrimitiveBuilder<Self::Arrow>, ExtendFromSliceResult = ()> + Debug + From<NativeType<Self>> + Into<NativeType<Self>>
{
    /// Equivalent Arrow primitive type
    type Arrow: ArrowPrimitiveType + Debug;
}
//
// Easy access to the ArrowPrimitiveType backing a PrimitiveType
type ArrowType<T> = <T as PrimitiveType>::Arrow;
//
// Easy access to the NativeType backing a PrimitiveType
pub(crate) type NativeType<T> = <ArrowType<T> as ArrowPrimitiveType>::Native;
//
// SAFETY: The types for which this trait is implemented must be a
//         `repr(transparent)` wrapper over the underlying
//         `ArrowPrimitiveType::NativeType`.
macro_rules! unsafe_impl_primitive_type {
    ($($local:ty => $arrow:ty),*) => {
        $(
            unsafe impl PrimitiveType for $local {
                type Arrow = $arrow;
            }

            // SAFETY: By construction, it is enforced that Slice is &[Self]
            unsafe impl ArrayElement for $local {
                type BuilderBackend = PrimitiveBuilder<ArrowType<$local>>;
                type WriteValue<'a> = Self;
                type ReadValue<'a> = Self;
                type WriteSlice<'a> = &'a [Self];
                type ReadSlice<'a> = &'a [Self];
                type ExtendFromSliceResult = ();
            }

            // SAFETY: PrimitiveBuilder does use a Bitmap validity slice
            unsafe impl OptionalElement for $local {
                type ValiditySlice<'a> = Bitmap<'a>;
            }

            impl_option_element!($local);
        )*
    };
}
//
// SAFETY: All types listed below are indeed repr(transparent) wrappers over the
//         corresponding arrow native type.
unsafe_impl_primitive_type!(
    Date32 => Date32Type,
    Date64 => Date64Type,
    // TODO: Support decimals, see above for rustc blocker info.
    Duration<Microsecond> => DurationMicrosecondType,
    Duration<Millisecond> => DurationMillisecondType,
    Duration<Nanosecond> => DurationNanosecondType,
    Duration<Second> => DurationSecondType,
    f16 => Float16Type,
    f32 => Float32Type,
    f64 => Float64Type,
    i8 => Int8Type,
    i16 => Int16Type,
    i32 => Int32Type,
    i64 => Int64Type,
    IntervalDayTime => IntervalDayTimeType,
    IntervalMonthDayNano => IntervalMonthDayNanoType,
    IntervalYearMonth => IntervalYearMonthType,
    Time<Millisecond> => Time32MillisecondType,
    Time<Second> => Time32SecondType,
    Time<Microsecond> => Time64MicrosecondType,
    Time<Nanosecond> => Time64NanosecondType,
    // TODO: Support timestamps, see above for rustc blocker info.
    u8 => UInt8Type,
    u16 => UInt16Type,
    u32 => UInt32Type,
    u64 => UInt64Type
);
