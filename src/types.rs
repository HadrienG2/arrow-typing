//! Strongly typed versions of arrow's DataTypes

use arrow_array::types::IntervalDayTimeType;
use std::{fmt::Debug, marker::PhantomData, num::TryFromIntError};

/// Date type representing the elapsed time since the UNIX epoch in days
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Date<T>(pub T);
pub type Date32 = Date<i32>;
pub type Date64 = Date<i64>;

// TODO: Need default const parameters to expose Decimal<T, PRECISION =
//       DEFAULT_PRECISION, SCALE = DEFAULT_SCALE>

/// Measure of elapsed time with a certain integer unit
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Duration<Unit: TimeUnit>(pub i64, PhantomData<Unit>);
type StdDuration = std::time::Duration;

/// Elapsed time since midnight
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Time<Unit: TimeUnit>(pub <Unit as TimeUnit>::TimeStorage);

// TODO: Need more flexibility in const parameters to represent timestamps
//       within an optional time zone.

/// Unit of time
pub trait TimeUnit {
    /// Storage format for time since midnight in this unit
    type TimeStorage: Clone + Copy + Debug;
}

/// Second duration storage granularity
pub struct Second;
//
impl TimeUnit for Second {
    type TimeStorage = i32;
}
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

/// Millisecond duration storage granularity
pub struct Millisecond;
//
impl TimeUnit for Millisecond {
    type TimeStorage = i32;
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

/// Microsecond duration storage granularity
pub struct Microsecond;
//
impl TimeUnit for Microsecond {
    type TimeStorage = i64;
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

/// Nanosecond duration storage granularity
pub struct Nanosecond;
//
impl TimeUnit for Nanosecond {
    type TimeStorage = i64;
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
