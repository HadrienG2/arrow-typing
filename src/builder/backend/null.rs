//! Strong typing layer on top of [`NullBuilder`]

use super::{Backend, Capacity, NoAlternateConfig, TypedBackend};
use crate::{
    builder::BuilderConfig,
    element::primitive::{ConstBoolSlice, Null, UniformSlice},
};
use arrow_array::builder::{ArrayBuilder, NullBuilder};
use arrow_schema::{DataType, Field};

impl Backend for NullBuilder {
    #[cfg(test)]
    fn capacity_opt(&self) -> Option<usize> {
        None
    }

    fn extend_with_nulls(&mut self, n: usize) {
        self.append_nulls(n)
    }

    type ValiditySlice<'a> = ConstBoolSlice<false>;

    fn validity_slice(&self) -> Option<ConstBoolSlice<false>> {
        Some(ConstBoolSlice::new(self.len()))
    }
}

impl Capacity for NullBuilder {
    fn capacity(&self) -> usize {
        usize::MAX
    }
}

impl TypedBackend<Null> for NullBuilder {
    type ExtraConfig = ();
    type AlternateConfig = NoAlternateConfig;

    fn make_field(_config: &BuilderConfig<Null>, name: String) -> Field {
        Field::new(name, DataType::Null, true)
    }

    fn new(_config: BuilderConfig<Null>) -> Self {
        Self::new()
    }

    #[inline]
    fn push(&mut self, _v: Null) {
        self.append_null()
    }

    fn extend_from_slice(&mut self, s: UniformSlice<Null>) {
        self.append_nulls(s.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::{
            tests::{
                check_extend_outcome, check_init_default, check_init_with_capacity_outcome,
                check_push,
            },
            TypedBuilder,
        },
        tests::length_or_capacity,
    };
    use proptest::{prelude::*, test_runner::TestCaseResult};

    #[test]
    fn init_default() -> TestCaseResult {
        check_init_default::<Null>()
    }

    proptest! {
        #[test]
        fn init_with_capacity(capacity in length_or_capacity()) {
            check_init_with_capacity_outcome(
                &TypedBuilder::<Null>::with_capacity(capacity),
                Some(capacity)
            )?;
        }

        #[test]
        fn push(init_capacity in length_or_capacity()) {
            check_push::<Null>(BuilderConfig::with_capacity(init_capacity), Null)?;
        }

        #[test]
        fn extend_with_nulls(
            init_capacity in length_or_capacity(),
            num_nulls in length_or_capacity()
        ) {
            let make_builder = || TypedBuilder::<Null>::with_capacity(init_capacity);
            {
                let mut builder = make_builder();
                builder.extend_from_slice(UniformSlice::new(Null, num_nulls));
                check_extend_outcome(&builder, init_capacity, num_nulls)?;
            }{
                let mut builder = make_builder();
                builder.extend_with_nulls(num_nulls);
                check_extend_outcome(&builder, init_capacity, num_nulls)?;
            }
        }
    }
}
