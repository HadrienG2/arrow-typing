//! Strong typing layer on top of [`NullBuilder`]

use super::{Backend, TypedBackend};
use crate::types::primitive::Null;
use arrow_array::builder::NullBuilder;

impl Backend for NullBuilder {
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

impl TypedBackend<Null> for NullBuilder {
    #[inline]
    fn push(&mut self, _v: Null) {
        self.append_null()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::{tests::check_init_default, TypedBuilder},
        tests::length_or_capacity,
        types::primitive::Null,
    };
    use proptest::{prelude::*, test_runner::TestCaseResult};

    #[test]
    fn init_default() -> TestCaseResult {
        check_init_default::<Null>()
    }

    // NullBuilder enforces len == capacity
    fn check_null_builder_len(builder: &TypedBuilder<Null>) -> TestCaseResult {
        prop_assert_eq!(builder.len(), builder.capacity());
        prop_assert_eq!(builder.is_empty(), builder.capacity() == 0);
        Ok(())
    }

    proptest! {
        #[test]
        fn init_with_capacity(capacity in length_or_capacity()) {
            // For null builders, len == capacity, which has... interesting
            // consequences
            let builder = TypedBuilder::<Null>::with_capacity((), capacity);
            prop_assert_eq!(builder.capacity(), capacity);
            check_null_builder_len(&builder)?;
        }

        #[test]
        fn push_null(init_capacity in length_or_capacity()) {
            let mut builder = TypedBuilder::<Null>::with_capacity((), init_capacity);
            builder.push(Null);
            prop_assert_eq!(builder.capacity(), init_capacity + 1);
            check_null_builder_len(&builder)?;
        }

        #[test]
        fn extend_with_nulls(
            init_capacity in length_or_capacity(),
            num_nulls in length_or_capacity()
        ) {
            let mut builder = TypedBuilder::<Null>::with_capacity((), init_capacity);
            builder.extend_with_nulls(num_nulls);
            prop_assert_eq!(builder.capacity(), init_capacity + num_nulls);
            check_null_builder_len(&builder)?;
        }
    }
}
