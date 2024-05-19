//! Strong typing layer on top of [`BooleanBuilder`]

use super::{Backend, Capacity, NoAlternateConfig, TypedBackend};
use crate::{
    bitmap::Bitmap,
    builder::BuilderConfig,
    element::option::{OptionReadSlice, OptionSlice, OptionWriteSlice},
};
use arrow_array::builder::{ArrayBuilder, BooleanBuilder};
use arrow_schema::{ArrowError, DataType, Field};

impl Backend for BooleanBuilder {
    #[cfg(test)]
    fn capacity_opt(&self) -> Option<usize> {
        Some(self.capacity())
    }

    fn extend_with_nulls(&mut self, n: usize) {
        self.append_nulls(n)
    }

    type ValiditySlice<'a> = Bitmap<'a>;

    fn option_validity_slice(&self) -> Option<Self::ValiditySlice<'_>> {
        self.validity_slice()
            .map(|validity| Bitmap::new(validity, self.len()))
    }
}

impl Capacity for BooleanBuilder {
    fn capacity(&self) -> usize {
        self.capacity()
    }
}

// Common parts of TypedBackend impls for bool and Option<bool>
macro_rules! typed_backend_common {
    ($element_type:ty, $is_option:literal) => {
        type ExtraConfig = ();
        type AlternateConfig = NoAlternateConfig;

        fn make_field(_config: &BuilderConfig<$element_type>, name: String) -> Field {
            Field::new(name, DataType::Boolean, $is_option)
        }

        fn new(config: BuilderConfig<$element_type>) -> Self {
            let BuilderConfig::Standard {
                capacity,
                extra: (),
            } = config
            else {
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

impl TypedBackend<bool> for BooleanBuilder {
    typed_backend_common!(bool, false);

    #[inline]
    fn push(&mut self, v: bool) {
        self.append_value(v)
    }

    fn extend_from_slice(&mut self, s: &[bool]) {
        self.append_slice(s)
    }

    fn as_slice(&self) -> Bitmap<'_> {
        Bitmap::new(self.values_slice(), self.len())
    }
}

impl TypedBackend<Option<bool>> for BooleanBuilder {
    typed_backend_common!(Option<bool>, true);

    #[inline]
    fn push(&mut self, v: Option<bool>) {
        self.append_option(v)
    }

    fn extend_from_slice(&mut self, slice: OptionWriteSlice<'_, bool>) -> Result<(), ArrowError> {
        self.append_values(slice.values, slice.is_valid)
    }

    fn as_slice(&self) -> OptionReadSlice<'_, bool> {
        OptionSlice {
            is_valid: self.optimized_validity_slice(),
            values: <Self as TypedBackend<bool>>::as_slice(self),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::tests::{
            check_extend_from_options, check_extend_from_values, check_extend_with_nulls,
            check_init_default_optional, check_init_with_capacity_optional, check_push, option_vec,
            options_eq,
        },
        element::option::OptionSlice,
        tests::length_or_capacity,
        BuilderConfig,
    };
    use proptest::{prelude::*, test_runner::TestCaseResult};

    fn eq(x: bool, y: bool) -> bool {
        x == y
    }

    #[test]
    fn init_default() -> TestCaseResult {
        check_init_default_optional::<bool>()
    }

    proptest! {
        #[test]
        fn init_with_capacity(capacity in length_or_capacity()) {
            check_init_with_capacity_optional::<bool>(
                || (),
                capacity
            )?;
        }

        #[test]
        fn push_value(init_capacity in length_or_capacity(), value: bool) {
            check_push::<bool>(
                BuilderConfig::with_capacity(init_capacity),
                value,
                eq
            )?;
        }

        #[test]
        fn push_option(init_capacity in length_or_capacity(), value: Option<bool>) {
            check_push::<Option<bool>>(
                BuilderConfig::with_capacity(init_capacity),
                value,
                options_eq::<bool>(eq)
            )?;
        }

        #[test]
        fn extend_from_values(init_capacity in length_or_capacity(), values: Vec<bool>) {
            check_extend_from_values::<bool>(
                || BuilderConfig::with_capacity(init_capacity),
                &values,
                eq
            )?;
        }

        #[test]
        fn extend_from_options(
            init_capacity in length_or_capacity(),
            (values, is_valid) in option_vec::<bool>(),
        ) {
            check_extend_from_options::<bool>(
                BuilderConfig::with_capacity(init_capacity),
                OptionSlice {
                    values: &values[..],
                    is_valid: &is_valid[..],
                },
                eq
            )?;
        }

        #[test]
        fn extend_with_nulls(
            init_capacity in length_or_capacity(),
            num_nulls in length_or_capacity()
        ) {
            check_extend_with_nulls::<bool>(
                BuilderConfig::with_capacity(init_capacity),
                num_nulls,
                eq
            )?;
        }
    }
}
