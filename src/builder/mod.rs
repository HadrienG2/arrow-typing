//! Strongly typed array builder

pub(crate) mod backend;

use self::backend::{Backend, ExtendFromSlice, TypedBackend};
use crate::{ArrayElement, NullableElement, SliceElement};
use arrow_array::builder::ArrayBuilder;
use std::fmt::Debug;

/// Strongly typed array builder
#[derive(Debug)]
pub struct TypedBuilder<T: ArrayElement>(BuilderBackend<T>);
//
impl<T: ArrayElement> TypedBuilder<T> {
    /// Create a new builder
    pub fn new(params: ConstructorParameters<T>) -> Self {
        Self(BuilderBackend::<T>::new(params))
    }

    /// Create a new builder with space for `capacity` elements
    pub fn with_capacity(params: ConstructorParameters<T>, capacity: usize) -> Self {
        Self(BuilderBackend::<T>::with_capacity(params, capacity))
    }

    /// Number of elements the array can hold without reallocating
    ///
    /// In the case of types that are internally stored as multiple columnar
    /// buffers, like structs or unions, a lower bound on the capacity of all
    /// underlying columns is returned.
    ///
    /// In the case of arrays of lists, capacity is to be understood as the
    /// number of sublists that the array can hold, not the cumulative number of
    /// elements across all sublists.
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// Append a single value into the builder
    ///
    /// For types with a complex internal structure, such row-wise insertion may
    /// be inefficient. Therefore, if you intend to insert many values, it is
    /// advised that you do not do so by calling this method in a loop, but
    /// instead look into the bulk insertion methods below.
    #[inline]
    pub fn push(&mut self, value: T::Value<'_>) {
        self.0.push(value)
    }

    /// Efficiently append multiple values into the builder
    ///
    /// This operation is available for all element types that implement
    /// [`SliceElement`]. See the documentation of this trait for more
    /// information on what you can expect from `T::Slice` and
    /// `T::ExtendFromSliceResult`.
    pub fn extend_from_slice(&mut self, s: T::Slice<'_>) -> T::ExtendFromSliceResult
    where
        T: SliceElement,
        // FIXME: Remove useless bound once SliceElement gets the required
        //        Backend: ExtendFromSlice bound
        T::BuilderBackend: ExtendFromSlice<T>,
    {
        self.0.extend_from_slice(s)
    }
}
//
impl<T: SliceElement> TypedBuilder<Option<T>>
where
    Option<T>: ArrayElement<BuilderBackend = BuilderBackend<T>>,
    // FIXME: Remove useless bound once SliceElement gets the required Backend:
    //        ExtendFromSlice bound
    T::BuilderBackend: ExtendFromSlice<T>,
{
    /// Efficiently append multiple non-null values into the builder
    ///
    /// This operation is available for every `TypedBuilder` of `Option<T>`
    /// where `T` is a [`SliceElement`]. Given a slice of `T`, it lets you do
    /// the optimized equivalent of calling `push(Some(value))` in a loop for
    /// each value inside of the slice.
    pub fn extend_from_value_slice(&mut self, vs: T::Slice<'_>) -> T::ExtendFromSliceResult {
        self.0.extend_from_slice(vs)
    }
}
//
impl<T: ArrayElement> TypedBuilder<T> {
    /// Efficiently append multiple null values into the builder
    ///
    /// This operation is available when T is a [nullable
    /// type](NullableElement), i.e. `Null` or `Option<T>`.
    pub fn extend_with_nulls(&mut self, n: usize)
    where
        T: NullableElement,
    {
        self.0.extend_with_nulls(n)
    }

    /// Number of elements that were appended into this builder
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Truth that no elements were appended into this builder
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    // TODO: Type-safe access to the validity bitmap for builders that have it?

    // TODO: Some equivalent of ArrayBuilder::finish() and finish_cloned that
    //       returns a TypedArrayRef
}
//
impl<T: ArrayElement> Default for TypedBuilder<T>
where
    ConstructorParameters<T>: Default,
{
    fn default() -> Self {
        Self(T::BuilderBackend::new(Default::default()))
    }
}
//
impl<'a, T: ArrayElement> Extend<T::Value<'a>> for TypedBuilder<T> {
    fn extend<I: IntoIterator<Item = T::Value<'a>>>(&mut self, iter: I) {
        for item in iter {
            self.push(item)
        }
    }
}

/// Parameters needed to construct an array of Ts
///
/// Arrays of simple types can be built with no extra information. For these
/// array types the constructor parameters are a simple `()` unit value, and
/// `TypedBuilder` implements `Default` as an alternative to the slightly
/// awkward `new(())` parameter-less constructor.
///
/// However, more advanced array types need constructor parameters. For example,
/// arrays of fixed-sized lists need an inner sublist size. In this case, those
/// constructor parameters end up forwarded into the `ConstructorParameters`
/// that are passed to `new()`, and `TypedBuilder` will not implement `Default`.
pub type ConstructorParameters<T> =
    <<T as ArrayElement>::BuilderBackend as Backend>::ConstructorParameters;

/// Shortcut to the arrow builder type used to construct an array of Ts
type BuilderBackend<T> = <T as ArrayElement>::BuilderBackend;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{types::primitive::Null, OptionSlice};
    use proptest::{prelude::*, sample::SizeRange, test_runner::TestCaseResult};

    /// Maximum array length/capacity used in unit tests
    const MAX_CAPACITY: usize = 256;

    /// Generate a capacity between 0 and MAX_CAPACITY
    fn length_or_capacity() -> impl Strategy<Value = usize> {
        0..=MAX_CAPACITY
    }

    /// Check outcome of initializing a TypedBuilder with some capacity
    fn check_init_with_capacity(
        builder: &TypedBuilder<impl ArrayElement>,
        capacity: usize,
    ) -> TestCaseResult {
        prop_assert!(builder.capacity() >= capacity);
        prop_assert_eq!(builder.len(), 0);
        prop_assert!(builder.is_empty());
        // TODO: Build and check final array
        Ok(())
    }

    /// Check outcome of initializing a TypedBuilder with the default capacity
    fn check_init_default<T: ArrayElement>() -> TestCaseResult
    where
        ConstructorParameters<T>: Default,
    {
        let mut builder = TypedBuilder::<T>::new(Default::default());
        check_init_with_capacity(&builder, builder.capacity())?;
        builder = TypedBuilder::<T>::default();
        check_init_with_capacity(&builder, builder.capacity())
    }

    #[test]
    fn init_default() -> TestCaseResult {
        check_init_default::<Null>()?;
        check_init_default::<bool>()?;
        check_init_default::<Option<bool>>()?;
        Ok(())
    }

    proptest! {
        #[test]
        fn init_with_capacity(capacity in length_or_capacity()) {
            // Null builders have an interesting notion of length + capacity
            let null_builder = TypedBuilder::<Null>::with_capacity((), capacity);
            prop_assert_eq!(null_builder.capacity(), capacity);
            prop_assert_eq!(null_builder.len(), capacity);
            prop_assert_eq!(null_builder.is_empty(), capacity == 0);

            // Other builders behave more like a Vec would
            check_init_with_capacity(
                &TypedBuilder::<bool>::with_capacity((), capacity),
                capacity
            )?;
            check_init_with_capacity(
                &TypedBuilder::<Option<bool>>::with_capacity((), capacity),
                capacity
            )?;
        }
    }

    /// Check outcome of inserting N values into a newly created TypedBuilder
    ///
    /// This does not work as expected on `TypedBuilder<Null>` because the
    /// notion of length/capacity used by the underlying `NullBuilder` is weird.
    fn check_extend_outcome(
        builder: &TypedBuilder<impl ArrayElement>,
        init_capacity: usize,
        num_elements: usize,
    ) -> TestCaseResult {
        prop_assert!(builder.capacity() >= init_capacity.max(num_elements));
        prop_assert_eq!(builder.len(), num_elements);
        prop_assert_eq!(builder.is_empty(), num_elements == 0);
        // TODO: Build and check final array
        Ok(())
    }

    /// Check outcome of pushing a value into a newly created TypedBuilder
    fn check_push<T: ArrayElement>(
        init_params: ConstructorParameters<T>,
        init_capacity: usize,
        value: T::Value<'_>,
    ) -> TestCaseResult {
        let mut builder = TypedBuilder::<T>::with_capacity(init_params, init_capacity);
        builder.push(value);
        check_extend_outcome(&builder, init_capacity, 1)
    }

    proptest! {
        #[test]
        fn push_null(init_capacity in length_or_capacity()) {
            let mut builder = TypedBuilder::<Null>::with_capacity((), init_capacity);
            builder.push(Null);
            prop_assert_eq!(builder.capacity(), init_capacity + 1);
            prop_assert_eq!(builder.len(), builder.capacity());
            prop_assert!(!builder.is_empty());
        }

        #[test]
        fn push_bool(init_capacity in length_or_capacity(), value: bool) {
            check_push::<bool>((), init_capacity, value)?;
        }
        #[test]
        fn push_option_bool(init_capacity in length_or_capacity(), value: Option<bool>) {
            check_push::<Option<bool>>((), init_capacity, value)?;
        }
    }

    /// Generate building blocks for an OptionSlice<T>
    fn option_vec<T: SliceElement + Arbitrary>() -> impl Strategy<Value = (Vec<T>, Vec<bool>)> {
        prop_oneof![
            // Valid OptionSlice
            (0..=SizeRange::default().end_incl()).prop_flat_map(|len| {
                (
                    prop::collection::vec(any::<T>(), len),
                    prop::collection::vec(any::<bool>(), len),
                )
            }),
            any::<(Vec<T>, Vec<bool>)>()
        ]
    }

    proptest! {
        #[test]
        fn extend_from_values_bool(init_capacity in length_or_capacity(), values: Vec<bool>) {
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
        fn extend_from_options_bool(
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
            let mut null_builder = TypedBuilder::<Null>::with_capacity((), init_capacity);
            null_builder.extend_with_nulls(num_nulls);
            prop_assert!(null_builder.capacity() >= init_capacity.max(num_nulls));
            prop_assert_eq!(null_builder.len(), null_builder.capacity());
            prop_assert!(!null_builder.is_empty());

            let mut opt_builder = TypedBuilder::<Option<bool>>::with_capacity((), init_capacity);
            opt_builder.extend_with_nulls(num_nulls);
            check_extend_outcome(&opt_builder, init_capacity, num_nulls)?;
        }
    }
}
