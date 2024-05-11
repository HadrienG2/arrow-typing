//! Strong typing layer on top of [`PrimitiveBuilder`]

use super::{Backend, Capacity, NoAlternateConfig, TypedBackend, ValiditySlice};
use crate::{
    builder::BuilderConfig,
    element::{
        primitive::{NativeType, PrimitiveType},
        ArrayElement, OptionSlice,
    },
};
use arrow_array::{builder::PrimitiveBuilder, types::ArrowPrimitiveType};
use arrow_schema::{ArrowError, Field};
use std::{fmt::Debug, panic::AssertUnwindSafe};

impl<T: ArrowPrimitiveType + Debug> Backend for PrimitiveBuilder<T> {
    #[cfg(test)]
    fn capacity_opt(&self) -> Option<usize> {
        Some(self.capacity())
    }

    fn extend_with_nulls(&mut self, n: usize) {
        self.append_nulls(n)
    }
}

impl<T: ArrowPrimitiveType + Debug> Capacity for PrimitiveBuilder<T> {
    fn capacity(&self) -> usize {
        usize::MAX
    }
}

impl<T: ArrowPrimitiveType + Debug> ValiditySlice for PrimitiveBuilder<T> {
    fn validity_slice(&self) -> Option<&[u8]> {
        self.validity_slice()
    }
}

// Common parts of TypedBackend impls for T and Option<T>
macro_rules! typed_backend_common {
    ($element_type:ty, $is_option:literal) => {
        type ExtraConfig = ();
        type AlternateConfig = NoAlternateConfig;

        fn make_field(_config: &BuilderConfig<$element_type>, name: String) -> Field {
            // TODO: Once we start supporting Decimal and Timestamp, allow
            //       config to affect the choice of data type.
            Field::new(name, T::Arrow::DATA_TYPE, $is_option)
        }

        fn new(config: BuilderConfig<$element_type>) -> Self {
            let BuilderConfig::Standard { capacity, extra: _ } = config else {
                unreachable!()
            };
            if let Some(capacity) = capacity {
                Self::with_capacity(capacity)
            } else {
                Self::new()
            }
        }
    };
}

impl<T: PrimitiveType> TypedBackend<T> for PrimitiveBuilder<T::Arrow>
where
    // TODO: Remove this bound once the Rust trait system supports adding the
    //       appropriate bounds on PrimitiveType to let rustc figure out that
    //       T::WriteValue<'_> is just T for primitive types (and thus
    //       T::WriteValue must implement Into<NativeType<T>> per PrimitiveType
    //       definition)
    for<'a> T::WriteValue<'a>: Into<NativeType<T>>,
{
    typed_backend_common!(T, false);

    #[inline]
    fn push(&mut self, v: T::WriteValue<'_>) {
        self.append_value(v.into())
    }

    fn extend_from_slice(&mut self, s: T::WriteSlice<'_>) {
        // SAFETY: This transmute is safe because...
        //         - T::WriteSlice is &[T] for all primitive types
        //         - Primitive types are repr(transparent) wrappers over the
        //           corresponding Arrow native types, so it is safe to
        //           transmute &[T] into &[NativeType<T>].
        let native_slice =
            unsafe { std::mem::transmute_copy::<T::WriteSlice<'_>, &[NativeType<T>]>(&s) };
        self.append_slice(native_slice)
    }
}

impl<T: PrimitiveType> TypedBackend<Option<T>> for PrimitiveBuilder<T::Arrow>
where
    // TODO: Remove this bound once the Rust trait system supports adding the
    //       appropriate bounds on PrimitiveType to let rustc figure out that
    //       T::WriteValue<'_> is just T for primitive types (and thus
    //       T::WriteValue must implement Into<NativeType<T>> per PrimitiveType
    //       definition)
    for<'a> T::WriteValue<'a>: Into<NativeType<T>>,
    //
    // TODO: Remove these bounds once it becomes possible to blanket-impl
    //       ArrayElement for Option<T: PrimitiveType>, making them obvious.
    Option<T>: ArrayElement<ExtendFromSliceResult = Result<(), ArrowError>>,
    for<'a> <Option<T> as ArrayElement>::WriteValue<'a>: Into<Option<T::WriteValue<'a>>>,
    for<'a> <Option<T> as ArrayElement>::WriteSlice<'a>: Into<OptionSlice<'a, T>>,
{
    typed_backend_common!(Option<T>, true);

    #[inline]
    fn push(&mut self, v: <Option<T> as ArrayElement>::WriteValue<'_>) {
        let opt: Option<T::WriteValue<'_>> = v.into();
        let opt: Option<NativeType<T>> = opt.map(Into::into);
        self.append_option(opt)
    }

    fn extend_from_slice(
        &mut self,
        slice: <Option<T> as ArrayElement>::WriteSlice<'_>,
    ) -> Result<(), ArrowError> {
        let slice: OptionSlice<T> = slice.into();
        // SAFETY: This transmute is safe for the same reason as above
        let native_values = unsafe {
            std::mem::transmute_copy::<T::WriteSlice<'_>, &[NativeType<T>]>(&slice.values)
        };
        let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
            self.append_values(native_values, slice.is_valid)
        }));
        res.map_err(|_| {
            ArrowError::InvalidArgumentError("value and validity lengths must be equal".to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::{
            tests::{
                check_extend_from_options, check_extend_from_values, check_extend_with_nulls,
                check_init_default_optional, check_init_with_capacity_optional, check_push,
                check_push_option, option_vec,
            },
            BuilderConfig,
        },
        element::{
            primitive::{
                Date32, Date64, Duration, IntervalDayTime, IntervalMonthDayNano, IntervalYearMonth,
                Microsecond, Millisecond, Nanosecond, Second, Time,
            },
            OptionSlice,
        },
        tests::length_or_capacity,
    };
    use proptest::{prelude::*, test_runner::TestCaseResult};

    macro_rules! test_primitives {
        ($primitive: ident) => {
            test_primitives!($primitive : $primitive);
        };
        ($mod_name:ident : $primitive:ty) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn init_default() -> TestCaseResult {
                    check_init_default_optional::<$primitive>()
                }

                proptest! {
                    #[test]
                    fn init_with_capacity(capacity in length_or_capacity()) {
                        check_init_with_capacity_optional::<$primitive>(|| (), capacity)?;
                    }

                    #[test]
                    fn push_value(init_capacity in length_or_capacity(), value: $primitive) {
                        check_push::<$primitive>(BuilderConfig::with_capacity(init_capacity), value)?;
                    }

                    #[test]
                    fn push_option(init_capacity in length_or_capacity(), value: Option<$primitive>) {
                        check_push_option::<$primitive>(BuilderConfig::with_capacity(init_capacity), value)?;
                    }

                    #[test]
                    fn extend_from_values(init_capacity in length_or_capacity(), values: Vec<$primitive>) {
                        check_extend_from_values::<$primitive>(
                            || BuilderConfig::with_capacity(init_capacity),
                            &values
                        )?;
                    }

                    #[test]
                    fn extend_from_options(
                        init_capacity in length_or_capacity(),
                        (values, is_valid) in option_vec::<$primitive>(),
                    ) {
                        check_extend_from_options::<$primitive>(
                            BuilderConfig::with_capacity(init_capacity),
                            OptionSlice {
                                values: &values,
                                is_valid: &is_valid,
                            }
                        )?;
                    }

                    #[test]
                    fn extend_with_nulls(
                        init_capacity in length_or_capacity(),
                        num_nulls in length_or_capacity()
                    ) {
                        check_extend_with_nulls::<$primitive>(BuilderConfig::with_capacity(init_capacity), num_nulls)?;
                    }
                }
            }
        };
        ($( $mod_name:ident $(: $primitive:ty)? ),*) => {$(
            test_primitives!($mod_name $(: $primitive)? );
        )*};
    }
    test_primitives!(
        date32: Date32,
        date64: Date64,
        duration_micros: Duration<Microsecond>,
        duration_millis: Duration<Millisecond>,
        duration_nanos: Duration<Nanosecond>,
        duration_secs: Duration<Second>,
        // TODO: Put f16 here once it implements Arbitrary
        f32, f64, i8, i16, i32, i64,
        interval_day_time: IntervalDayTime,
        interval_month_day_nano: IntervalMonthDayNano,
        interval_year_month: IntervalYearMonth,
        time_micros: Time<Microsecond>,
        time_millis: Time<Millisecond>,
        time_nanos: Time<Nanosecond>,
        time_secs: Time<Second>,
        u8, u16, u32, u64
    );

    // TODO: Since f16 does not implement Arbitrary yet, it cannot leverage the
    //       above test macro and needs a custom test harness. See
    //       https://github.com/starkat99/half-rs/issues/110 .
    mod f16 {
        use super::*;
        use crate::builder::tests::option_vec_custom;
        use half::f16;
        use proptest::sample::SizeRange;

        fn any_f16() -> impl Strategy<Value = f16> {
            any::<u16>().prop_map(f16::from_bits)
        }

        fn any_f16_opt() -> impl Strategy<Value = Option<f16>> {
            prop_oneof![Just(None), any_f16().prop_map(Some)]
        }

        fn any_f16_vec() -> impl Strategy<Value = Vec<f16>> {
            prop::collection::vec(any_f16(), SizeRange::default())
        }

        #[test]
        fn init_default() -> TestCaseResult {
            check_init_default_optional::<f16>()
        }

        proptest! {
            #[test]
            fn init_with_capacity(capacity in length_or_capacity()) {
                check_init_with_capacity_optional::<f16>(|| (), capacity)?;
            }

            #[test]
            fn push_value(init_capacity in length_or_capacity(), value in any_f16()) {
                check_push::<f16>(BuilderConfig::with_capacity(init_capacity), value)?;
            }

            #[test]
            fn push_option(init_capacity in length_or_capacity(), value in any_f16_opt()) {
                check_push_option::<f16>(BuilderConfig::with_capacity(init_capacity), value)?;
            }

            #[test]
            fn extend_from_values(init_capacity in length_or_capacity(), values in any_f16_vec()) {
                check_extend_from_values::<f16>(
                    || BuilderConfig::with_capacity(init_capacity),
                    &values
                )?;
            }

            #[test]
            fn extend_from_options(
                init_capacity in length_or_capacity(),
                (values, is_valid) in option_vec_custom(any_f16),
            ) {
                check_extend_from_options::<f16>(
                    BuilderConfig::with_capacity(init_capacity),
                    OptionSlice {
                        values: &values,
                        is_valid: &is_valid,
                    }
                )?;
            }

            #[test]
            fn extend_with_nulls(
                init_capacity in length_or_capacity(),
                num_nulls in length_or_capacity()
            ) {
                check_extend_with_nulls::<f16>(BuilderConfig::with_capacity(init_capacity), num_nulls)?;
            }
        }
    }
}
