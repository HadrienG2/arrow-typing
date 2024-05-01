//! Strong typing layer on top of [`BooleanBuilder`]

use super::{Backend, ExtendFromSlice, TypedBackend};
use crate::OptionSlice;
use arrow_array::builder::BooleanBuilder;
use arrow_schema::ArrowError;

impl Backend for BooleanBuilder {
    type ConstructorParameters = ();

    fn new(_params: ()) -> Self {
        Self::new()
    }

    fn with_capacity(_params: (), capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }

    fn capacity(&self) -> usize {
        self.capacity()
    }

    fn extend_with_nulls(&mut self, n: usize) {
        self.append_nulls(n)
    }
}

impl TypedBackend<bool> for BooleanBuilder {
    #[inline]
    fn push(&mut self, v: bool) {
        self.append_value(v)
    }
}

impl TypedBackend<Option<bool>> for BooleanBuilder {
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
        builder::{
            tests::{
                check_extend_outcome, check_init_default, check_init_with_capacity, check_push,
                option_vec,
            },
            TypedBuilder,
        },
        tests::length_or_capacity,
        OptionSlice,
    };
    use proptest::{prelude::*, test_runner::TestCaseResult};

    #[test]
    fn init_default() -> TestCaseResult {
        check_init_default::<bool>()?;
        check_init_default::<Option<bool>>()?;
        Ok(())
    }

    proptest! {
        #[test]
        fn init_with_capacity(capacity in length_or_capacity()) {
            check_init_with_capacity(
                &TypedBuilder::<bool>::with_capacity((), capacity),
                capacity
            )?;
            check_init_with_capacity(
                &TypedBuilder::<Option<bool>>::with_capacity((), capacity),
                capacity
            )?;
        }

        #[test]
        fn push_value(init_capacity in length_or_capacity(), value: bool) {
            check_push::<bool>((), init_capacity, value)?;
        }
        #[test]
        fn push_option(init_capacity in length_or_capacity(), value: Option<bool>) {
            check_push::<Option<bool>>((), init_capacity, value)?;
        }

        #[test]
        fn extend_from_values(init_capacity in length_or_capacity(), values: Vec<bool>) {
            let bool_builder = || TypedBuilder::<bool>::with_capacity((), init_capacity);
            {
                let mut bool_builder = bool_builder();
                bool_builder.extend_from_slice(&values);
                check_extend_outcome(&bool_builder, init_capacity, values.len())?;
            }
            {
                let mut bool_builder = bool_builder();
                bool_builder.extend(values.iter().copied());
                check_extend_outcome(&bool_builder, init_capacity, values.len())?;
            }

            let opt_builder = || TypedBuilder::<Option<bool>>::with_capacity((), init_capacity);
            {
                let mut opt_builder = opt_builder();
                opt_builder.extend_from_value_slice(&values);
                check_extend_outcome(&opt_builder, init_capacity, values.len())?;
            }
            {
                let mut opt_builder = opt_builder();
                opt_builder.extend(values.iter().map(|&b| Some(b)));
                check_extend_outcome(&opt_builder, init_capacity, values.len())?;
            }
        }

        #[test]
        fn extend_from_options(
            init_capacity in length_or_capacity(),
            (values, is_valid) in option_vec::<bool>(),
        ) {
            let mut builder = TypedBuilder::<Option<bool>>::with_capacity((), init_capacity);
            let result = builder.extend_from_slice(OptionSlice {
                values: &values,
                is_valid: &is_valid,
            });

            if values.len() != is_valid.len() {
                prop_assert!(result.is_err());
                check_init_with_capacity(&builder, init_capacity)?;
                return Ok(());
            }

            prop_assert!(result.is_ok());
            check_extend_outcome(&builder, init_capacity, values.len())?;
        }

        #[test]
        fn extend_with_nulls(
            init_capacity in length_or_capacity(),
            num_nulls in length_or_capacity()
        ) {
            let mut opt_builder = TypedBuilder::<Option<bool>>::with_capacity((), init_capacity);
            opt_builder.extend_with_nulls(num_nulls);
            check_extend_outcome(&opt_builder, init_capacity, num_nulls)?;
        }
    }
}
