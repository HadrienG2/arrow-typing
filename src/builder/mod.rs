//! Strongly typed array builder

pub(crate) mod backend;

use self::backend::{Backend, ExtendFromSlice, TypedBackend};
#[cfg(doc)]
use crate::types::primitive::PrimitiveType;
use crate::{validity::ValiditySlice, ArrayElement, NullableElement, SliceElement};
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

    // TODO: Some equivalent of ArrayBuilder::finish() and finish_cloned that
    //       returns a TypedArrayRef
}
//
impl<T> TypedBuilder<Option<T>>
where
    Option<T>: ArrayElement,
    BuilderBackend<Option<T>>: backend::ValiditySlice,
{
    /// Current null buffer / validity slice
    ///
    /// This operation is only available on `TypedBuilder`s of optional `bool`s,
    /// [primitive types](PrimitiveType), bytes and strings.
    pub fn validity_slice(&self) -> Option<ValiditySlice<'_>> {
        use backend::ValiditySlice;
        self.0
            .validity_slice()
            .map(|bitmap| crate::validity::ValiditySlice::new(bitmap, self.len()))
    }
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
    use proptest::{prelude::*, sample::SizeRange, test_runner::TestCaseResult};

    /// Check outcome of initializing a TypedBuilder with some capacity
    ///
    /// This does not work with NullBuilder, for which len == capacity
    pub fn check_init_with_capacity(
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
    pub fn check_init_default<T: ArrayElement>() -> TestCaseResult
    where
        ConstructorParameters<T>: Default,
    {
        let mut builder = TypedBuilder::<T>::new(Default::default());
        check_init_with_capacity(&builder, builder.capacity())?;
        builder = TypedBuilder::<T>::default();
        check_init_with_capacity(&builder, builder.capacity())
    }

    /// Check outcome of inserting N values into a newly created TypedBuilder
    ///
    /// This does not work as expected on `TypedBuilder<Null>` because the
    /// notion of length/capacity used by the underlying `NullBuilder` is weird.
    pub fn check_extend_outcome(
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
    pub fn check_push<T: ArrayElement>(
        init_params: ConstructorParameters<T>,
        init_capacity: usize,
        value: T::Value<'_>,
    ) -> TestCaseResult {
        let mut builder = TypedBuilder::<T>::with_capacity(init_params, init_capacity);
        builder.push(value);
        check_extend_outcome(&builder, init_capacity, 1)
    }

    /// Generate building blocks for an OptionSlice<T>
    pub fn option_vec<T: SliceElement + Arbitrary>() -> impl Strategy<Value = (Vec<T>, Vec<bool>)> {
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
}
