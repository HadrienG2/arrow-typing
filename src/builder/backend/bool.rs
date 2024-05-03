//! Strong typing layer on top of [`BooleanBuilder`]

use super::{Backend, ExtendFromSlice, TypedBackend, ValiditySlice};
use crate::{builder::BuilderConfig, OptionSlice};
use arrow_array::builder::BooleanBuilder;
use arrow_schema::ArrowError;

impl Backend for BooleanBuilder {
    fn capacity(&self) -> usize {
        self.capacity()
    }

    fn extend_with_nulls(&mut self, n: usize) {
        self.append_nulls(n)
    }
}

impl ValiditySlice for BooleanBuilder {
    fn validity_slice(&self) -> Option<&[u8]> {
        self.validity_slice()
    }
}

impl TypedBackend<bool> for BooleanBuilder {
    type Config = ();

    fn new(params: BuilderConfig<bool>) -> Self {
        if let Some(capacity) = params.capacity {
            Self::with_capacity(capacity)
        } else {
            Self::new()
        }
    }

    #[inline]
    fn push(&mut self, v: bool) {
        self.append_value(v)
    }
}

impl TypedBackend<Option<bool>> for BooleanBuilder {
    type Config = ();

    fn new(params: BuilderConfig<Option<bool>>) -> Self {
        if let Some(capacity) = params.capacity {
            Self::with_capacity(capacity)
        } else {
            Self::new()
        }
    }

    #[inline]
    fn push(&mut self, v: Option<bool>) {
        self.append_option(v)
    }
}

impl ExtendFromSlice<bool> for BooleanBuilder {
    fn extend_from_slice(&mut self, s: &[bool]) {
        self.append_slice(s)
    }
}

impl ExtendFromSlice<Option<bool>> for BooleanBuilder {
    fn extend_from_slice(&mut self, slice: OptionSlice<'_, bool>) -> Result<(), ArrowError> {
        self.append_values(slice.values, slice.is_valid)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::tests::{
            check_extend_from_options, check_extend_from_values, check_extend_with_nulls,
            check_init_default_optional, check_init_with_capacity_optional, check_push,
            check_push_option, option_vec,
        },
        tests::length_or_capacity,
        OptionSlice,
    };
    use proptest::{prelude::*, test_runner::TestCaseResult};

    #[test]
    fn init_default() -> TestCaseResult {
        check_init_default_optional::<bool>()
    }

    proptest! {
        #[test]
        fn init_with_capacity(capacity in length_or_capacity()) {
            check_init_with_capacity_optional::<bool>(|| (), capacity)?;
        }

        #[test]
        fn push_value(init_capacity in length_or_capacity(), value: bool) {
            check_push::<bool>((), init_capacity, value)?;
        }

        #[test]
        fn push_option(init_capacity in length_or_capacity(), value: Option<bool>) {
            check_push_option::<bool>((), init_capacity, value)?;
        }

        #[test]
        fn extend_from_values(init_capacity in length_or_capacity(), values: Vec<bool>) {
            check_extend_from_values::<bool>(|| (), init_capacity, &values)?;
        }

        #[test]
        fn extend_from_options(
            init_capacity in length_or_capacity(),
            (values, is_valid) in option_vec::<bool>(),
        ) {
            check_extend_from_options::<bool>((), init_capacity, OptionSlice {
                values: &values,
                is_valid: &is_valid,
            })?;
        }

        #[test]
        fn extend_with_nulls(
            init_capacity in length_or_capacity(),
            num_nulls in length_or_capacity()
        ) {
            check_extend_with_nulls::<bool>((), init_capacity, num_nulls)?;
        }
    }
}
