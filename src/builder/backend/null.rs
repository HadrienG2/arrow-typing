//! Strong typing layer on top of [`NullBuilder`]

use super::{Backend, TypedBackend};
use crate::{builder::BuilderConfig, types::primitive::Null};
use arrow_array::builder::NullBuilder;

impl Backend for NullBuilder {
    fn capacity(&self) -> usize {
        usize::MAX
    }

    fn extend_with_nulls(&mut self, n: usize) {
        self.append_nulls(n)
    }
}

impl TypedBackend<Null> for NullBuilder {
    type Config = ();

    fn new(_config: BuilderConfig<Null>) -> Self {
        // FIXME: We do not forward the capacity to NullBuilder as it does not
        //        handle it in a manner that is consistent with other builders,
        //        see https://github.com/apache/arrow-rs/issues/5711
        Self::new()
    }

    #[inline]
    fn push(&mut self, _v: Null) {
        self.append_null()
    }

    fn extend_from_slice(&mut self, n: usize) {
        self.append_nulls(n)
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
                capacity
            )?;
        }

        #[test]
        fn push(init_capacity in length_or_capacity()) {
            check_push::<Null>((), init_capacity, Null)?;
        }

        #[test]
        fn extend_with_nulls(
            init_capacity in length_or_capacity(),
            num_nulls in length_or_capacity()
        ) {
            let make_builder = || TypedBuilder::<Null>::with_capacity(init_capacity);
            {
                let mut builder = make_builder();
                builder.extend_from_slice(num_nulls);
                check_extend_outcome(&builder, init_capacity, num_nulls)?;
            }{
                let mut builder = make_builder();
                builder.extend_with_nulls(num_nulls);
                check_extend_outcome(&builder, init_capacity, num_nulls)?;
            }
        }
    }
}
