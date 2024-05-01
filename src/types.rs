//! Strongly typed versions of arrow's DataTypes

use crate::{ArrayElement, NullableElement, SliceElement};
use arrow_array::{builder::NullBuilder, types::*};
use half::f16;
use std::{fmt::Debug, marker::PhantomData, num::TryFromIntError};

// === Strong value types matching non-std Arrow DataTypes ===

/// A value that is always null
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Null;
//
impl ArrayElement for Null {
    type BuilderBackend = NullBuilder;
    type Value<'a> = Self;
}
//
impl NullableElement for Null {}

/// Date type representing the elapsed time since the UNIX epoch in days
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Date<T>(T);
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

// TODO: Need default const parameters to expose Decimal<T, const
//       STORAGE_PRECISION = DEFAULT_PRECISION, const STORAGE_SCALE =
//       DEFAULT_SCALE>

/// Measure of elapsed time with a certain integer unit
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Duration<Unit: TimeUnit>(i64, PhantomData<Unit>);
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
#[derive(Clone, Copy, Debug)]
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
#[derive(Clone, Copy, Debug)]
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
#[derive(Clone, Copy, Debug)]
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
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Time<Unit: TimeUnit>(<Unit as TimeUnit>::TimeStorage);
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

// TODO: Need more flexibility in const parameters to cleanly represent
//       Timestamp<Unit, const TIMESTAMP: Option<Arc<str>>> for optional
//       timezone

/// Unit of time
pub trait TimeUnit {
    /// Storage format for time since midnight in this unit
    type TimeStorage: Clone + Copy + Debug;
}

/// Second duration storage granularity
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Second;
//
impl TimeUnit for Second {
    type TimeStorage = i32;
}

/// Millisecond duration storage granularity
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Millisecond;
//
impl TimeUnit for Millisecond {
    type TimeStorage = i32;
}

/// Microsecond duration storage granularity
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Microsecond;
//
impl TimeUnit for Microsecond {
    type TimeStorage = i64;
}

/// Nanosecond duration storage granularity
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Nanosecond;
//
impl TimeUnit for Nanosecond {
    type TimeStorage = i64;
}

// === Equivalent of ArrowPrimitiveType for the types defined in this module ===

/// Mapping from the types defined in this module to [`ArrowPrimitiveType`]
///
/// # Safety
///
/// The type for which this trait is implemented must be a `repr(transparent)`
/// wrapper over the underlying `ArrowPrimitiveType::NativeType`.
pub unsafe trait PrimitiveType:
    // TODO: Once Rust's trait solver supports it, use a SliceElement<Value<'_>
    //       = Self> bound to simplify downstream usage.
    Debug + From<NativeType<Self>> + Into<NativeType<Self>> + SliceElement
{
    /// Equivalent Arrow primitive type
    type ArrowPrimitive: ArrowPrimitiveType + Debug;
}
//
macro_rules! unsafe_impl_primitive_type {
    ($($local:ty => $arrow:ty),*) => {
        $(
            unsafe impl PrimitiveType for $local {
                type ArrowPrimitive = $arrow;
            }
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

// Easy access to the NativeType backing a PrimitiveType
pub(crate) type NativeType<T> =
    <<T as PrimitiveType>::ArrowPrimitive as ArrowPrimitiveType>::Native;
