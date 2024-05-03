//! Strongly typed array builder

pub(crate) mod backend;

use self::backend::{Backend, ExtendFromSlice, TypedBackend};
#[cfg(doc)]
use crate::types::primitive::PrimitiveType;
use crate::{validity::ValiditySlice, ArrayElement, NullableElement, SliceElement};
use arrow_array::builder::ArrayBuilder;
use std::fmt::{self, Debug, Formatter};

/// Strongly typed array builder
#[derive(Debug)]
pub struct TypedBuilder<T: ArrayElement + ?Sized>(BuilderBackend<T>);
//
impl<T: ArrayElement + ?Sized> TypedBuilder<T>
where
    BackendConfig<T>: Default,
{
    /// Create a new array builder with the default configuration
    ///
    /// This constructor is availble for simple element types like primitive
    /// types which require no extra configuration. More complex element types
    /// (e.g. fixed-sized lists of dynamically defined extent) will need to be
    /// configured using the [`TypedBuilder::with_config()`] constructor.
    pub fn new() -> Self {
        Self(BuilderBackend::<T>::new(BuilderConfig::<T>::new()))
    }

    /// Create a new array builder with space for `capacity` elements
    ///
    /// See remark on [`new()`] concerning availability of this constructor.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(BuilderBackend::<T>::new(BuilderConfig::<T>::with_capacity(
            capacity,
        )))
    }
}
//
impl<T: ArrayElement + ?Sized> TypedBuilder<T> {
    /// Create a new array builder with an explicit configuration
    pub fn with_config(config: BuilderConfig<T>) -> Self {
        Self(BuilderBackend::<T>::new(config))
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
    /// For types with a complex internal structure, such element-wise insertion
    /// may be inefficient. Therefore, if you intend to insert many values, it
    /// is advised that you do not do so by calling this method in a loop, but
    /// instead look into the bulk insertion methods below.
    #[inline]
    pub fn push(&mut self, value: T::Value<'_>) {
        self.0.push(value)
    }

    /// Efficiently append multiple values into the builder
    ///
    /// This operation is available for all element types that implement
    /// [`SliceElement`]. See the documentation of this trait for more
    /// information on what types you can expect for `T::Slice` and
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
impl<T: ArrayElement + ?Sized> TypedBuilder<T> {
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
impl<T: ArrayElement + ?Sized> Default for TypedBuilder<T>
where
    BackendConfig<T>: Default,
{
    fn default() -> Self {
        Self::new()
    }
}
//
impl<'a, T: ArrayElement + ?Sized> Extend<T::Value<'a>> for TypedBuilder<T> {
    fn extend<I: IntoIterator<Item = T::Value<'a>>>(&mut self, iter: I) {
        for item in iter {
            self.push(item)
        }
    }
}

/// Configuration needed to construct a [`TypedBuilder`]
#[derive()]
pub struct BuilderConfig<T: ArrayElement + ?Sized> {
    /// Minimal number of elements this builder can accept without reallocating
    capacity: Option<usize>,

    /// Backend-specific configuration
    backend: BackendConfig<T>,
}
//
impl<T: ArrayElement + ?Sized> BuilderConfig<T>
where
    BackendConfig<T>: Default,
{
    /// Default builder configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder configuration with allocated room for `capacity` elements
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: Some(capacity),
            backend: Default::default(),
        }
    }
}
//
impl<T: ArrayElement + ?Sized> Clone for BuilderConfig<T>
where
    BackendConfig<T>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            backend: self.backend.clone(),
        }
    }
}
//
impl<T: ArrayElement + ?Sized> Debug for BuilderConfig<T>
where
    BackendConfig<T>: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("BuilderConfig")
            .field("capacity", &self.capacity)
            .field("backend", &self.backend)
            .finish()
    }
}
//
impl<T: ArrayElement + ?Sized> Default for BuilderConfig<T>
where
    BackendConfig<T>: Default,
{
    fn default() -> Self {
        Self {
            capacity: None,
            backend: Default::default(),
        }
    }
}
//
impl<T: ArrayElement + ?Sized> Eq for BuilderConfig<T> where BackendConfig<T>: Eq {}
//
impl<T: ArrayElement + ?Sized> PartialEq for BuilderConfig<T>
where
    BackendConfig<T>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.capacity == other.capacity && self.backend == other.backend
    }
}

/// Shortcut to the arrow builder type used to construct an array of Ts
type BuilderBackend<T> = <T as ArrayElement>::BuilderBackend;

/// Array builder configuration that is specific to a given element type `T`
///
/// Arrays of simple types can be built with no extra configuration. For these
/// array types the backend configuration is a simple `()` unit value, and
/// `TypedBuilder` will provide simple `new()` and `with_capacity()`
/// constructors and implement `Default`.
///
/// Arrays of more advanced types, however, need configuration. For example,
/// arrays of fixed-sized lists where the list size is chosen at runtime need to
/// be configured with an inner sublist size. In this case, the
/// [`TypedBuilder::with_config()`] constructor must be used, and it will
/// directly or indirectly receive this configuration type as a parameter.
type BackendConfig<T> = <BuilderBackend<T> as TypedBackend<T>>::Config;

#[allow(private_bounds)]
#[cfg(test)]
mod tests {
    use crate::OptionSlice;

    use super::*;
    use arrow_schema::ArrowError;
    use backend::ValiditySlice;
    use proptest::{prelude::*, sample::SizeRange, test_runner::TestCaseResult};

    /// Check the validity mask of a TypedBuilder that has the validity_slice()
    /// extension
    pub fn check_validity<T>(builder: &TypedBuilder<Option<T>>, expected: &[bool]) -> TestCaseResult
    where
        Option<T>: ArrayElement,
        BuilderBackend<Option<T>>: ValiditySlice,
    {
        if let Some(validity_slice) = builder.validity_slice() {
            prop_assert_eq!(validity_slice, expected);
        } else {
            prop_assert!(expected.iter().all(|valid| *valid));
        }
        Ok(())
    }

    /// Check outcome of initializing a `TypedBuilder` with some capacity
    ///
    /// This does not work with `NullBuilder`, for which `len == capacity`
    pub fn check_init_with_capacity_outcome(
        builder: &TypedBuilder<impl ArrayElement>,
        capacity: usize,
    ) -> TestCaseResult {
        prop_assert!(builder.capacity() >= capacity);
        prop_assert_eq!(builder.len(), 0);
        prop_assert!(builder.is_empty());
        // TODO: Build and check final array
        Ok(())
    }

    /// Like `check_init_with_capacity`, but for both `T` and `Option<T>`
    ///
    /// For almost every [`ArrayElement`] type `T` with the exception of `Null`,
    /// `Option<T>` is also an `ArrayElement`.
    pub fn check_init_with_capacity_optional<T: ArrayElement>(
        make_backend_config: impl Fn() -> BackendConfig<T>,
        capacity: usize,
    ) -> TestCaseResult
    where
        Option<T>: ArrayElement<BuilderBackend = BuilderBackend<T>>,
        BuilderBackend<T>: TypedBackend<Option<T>, Config = BackendConfig<T>>,
    {
        check_init_with_capacity_outcome(
            &TypedBuilder::<T>::with_config(BuilderConfig {
                capacity: Some(capacity),
                backend: make_backend_config(),
            }),
            capacity,
        )?;
        check_init_with_capacity_outcome(
            &TypedBuilder::<Option<T>>::with_config(BuilderConfig {
                capacity: Some(capacity),
                backend: make_backend_config(),
            }),
            capacity,
        )?;
        Ok(())
    }

    /// Check outcome of initializing a `TypedBuilder` with the default capacity
    pub fn check_init_default<T: ArrayElement>() -> TestCaseResult
    where
        BackendConfig<T>: Default,
    {
        let mut builder = TypedBuilder::<T>::new();
        check_init_with_capacity_outcome(&builder, builder.capacity())?;
        builder = TypedBuilder::<T>::default();
        check_init_with_capacity_outcome(&builder, builder.capacity())?;
        Ok(())
    }

    /// Like `check_init_default`, but for both `T` and `Option<T>`
    ///
    /// For almost every [`ArrayElement`] type `T` with the exception of `Null`,
    /// `Option<T>` is also an `ArrayElement`.
    pub fn check_init_default_optional<T: ArrayElement>() -> TestCaseResult
    where
        Option<T>: ArrayElement,
        BackendConfig<T>: Default,
        BackendConfig<Option<T>>: Default,
    {
        check_init_default::<T>()?;
        check_init_default::<Option<T>>()?;
        Ok(())
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
        backend_config: BackendConfig<T>,
        init_capacity: usize,
        value: T::Value<'_>,
    ) -> TestCaseResult {
        let mut builder = TypedBuilder::<T>::with_config(BuilderConfig {
            capacity: Some(init_capacity),
            backend: backend_config,
        });
        builder.push(value);
        check_extend_outcome(&builder, init_capacity, 1)?;
        Ok(())
    }

    /// Like `check_push`, but with `Option<T>` and validity bitmap checking
    pub fn check_push_option<T: ArrayElement>(
        backend_config: BackendConfig<Option<T>>,
        init_capacity: usize,
        value: Option<T>,
    ) -> TestCaseResult
    where
        Option<T>: ArrayElement,
        BuilderBackend<Option<T>>: ValiditySlice,
        for<'a> Option<T>: Into<<Option<T> as ArrayElement>::Value<'a>>,
    {
        let mut builder = TypedBuilder::<Option<T>>::with_config(BuilderConfig {
            capacity: Some(init_capacity),
            backend: backend_config,
        });
        let valid = value.is_some();
        builder.push(value.into());
        check_extend_outcome(&builder, init_capacity, 1)?;
        check_validity(&builder, &[valid])?;
        Ok(())
    }

    /// Check outcome of extending a builder of T or Option<T> with a slice of
    /// values
    pub fn check_extend_from_values<T: SliceElement>(
        make_backend_config: impl Fn() -> BackendConfig<T>,
        init_capacity: usize,
        values: T::Slice<'_>,
    ) -> TestCaseResult
    where
        Option<T>: ArrayElement<BuilderBackend = BuilderBackend<T>>,
        BuilderBackend<T>: ExtendFromSlice<T> + TypedBackend<Option<T>, Config = BackendConfig<T>>,
        BuilderBackend<Option<T>>: ValiditySlice,
        for<'a> T::Slice<'a>: Slice<T::Value<'a>> + Clone,
        for<'a> T::Value<'a>: Clone + Into<<Option<T> as ArrayElement>::Value<'a>>,
    {
        let make_value_builder = || {
            TypedBuilder::<T>::with_config(BuilderConfig {
                capacity: Some(init_capacity),
                backend: make_backend_config(),
            })
        };
        {
            let mut value_builder = make_value_builder();
            value_builder.extend_from_slice(values.clone());
            check_extend_outcome(&value_builder, init_capacity, values.slice_len())?;
        }
        {
            let mut value_builder = make_value_builder();
            value_builder.extend(values.slice_iter().cloned());
            check_extend_outcome(&value_builder, init_capacity, values.slice_len())?;
        }

        let make_opt_builder = || {
            TypedBuilder::<Option<T>>::with_config(BuilderConfig {
                capacity: Some(init_capacity),
                backend: make_backend_config(),
            })
        };
        {
            let mut opt_builder = make_opt_builder();
            opt_builder.extend_from_value_slice(values.clone());
            check_extend_outcome(&opt_builder, init_capacity, values.slice_len())?;
            check_validity(&opt_builder, &vec![true; values.slice_len()])?;
        }
        {
            let mut opt_builder = make_opt_builder();
            opt_builder.extend(values.slice_iter().cloned().map(Into::into));
            check_extend_outcome(&opt_builder, init_capacity, values.slice_len())?;
            check_validity(&opt_builder, &vec![true; values.slice_len()])?;
        }
        Ok(())
    }
    //
    trait Slice<T>: Clone {
        fn slice_len(&self) -> usize;
        fn slice_iter<'self_>(&'self_ self) -> impl Iterator<Item = &T> + 'self_
        where
            T: 'self_;
    }
    //
    impl<T> Slice<T> for &[T] {
        fn slice_len(&self) -> usize {
            self.len()
        }
        fn slice_iter<'self_>(&'self_ self) -> impl Iterator<Item = &T> + 'self_
        where
            T: 'self_,
        {
            self.iter()
        }
    }

    /// Generate building blocks for an `OptionSlice<T>`
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

    /// Like `option_vec`, but with a custom value generation strategy
    pub fn option_vec_custom<T: SliceElement, S: Strategy<Value = T>>(
        strategy: impl Fn() -> S + Copy,
    ) -> impl Strategy<Value = (Vec<T>, Vec<bool>)> {
        prop_oneof![
            // Valid OptionSlice
            (0..=SizeRange::default().end_incl()).prop_flat_map(move |len| {
                (
                    prop::collection::vec(strategy(), len),
                    prop::collection::vec(any::<bool>(), len),
                )
            }),
            (
                prop::collection::vec(strategy(), SizeRange::default()),
                any::<Vec<bool>>(),
            )
        ]
    }

    /// Check `extend_from_slice` on `TypedBuilder<Option<T>>`.
    pub fn check_extend_from_options<T: SliceElement>(
        backend_config: BackendConfig<Option<T>>,
        init_capacity: usize,
        slice: OptionSlice<T>,
    ) -> TestCaseResult
    where
        Option<T>: SliceElement<ExtendFromSliceResult = Result<(), ArrowError>>,
        for<'a> T::Slice<'a>: Slice<T>,
        for<'a> OptionSlice<'a, T>: Into<<Option<T> as SliceElement>::Slice<'a>>,
        BuilderBackend<Option<T>>: ExtendFromSlice<Option<T>> + ValiditySlice,
    {
        let mut builder = TypedBuilder::<Option<T>>::with_config(BuilderConfig {
            capacity: Some(init_capacity),
            backend: backend_config,
        });
        let result = builder.extend_from_slice(slice.clone().into());

        if slice.values.slice_len() != slice.is_valid.len() {
            prop_assert!(result.is_err());
            check_init_with_capacity_outcome(&builder, init_capacity)?;
            return Ok(());
        }

        prop_assert!(result.is_ok());
        check_extend_outcome(&builder, init_capacity, slice.values.slice_len())?;
        check_validity(&builder, slice.is_valid)?;
        Ok(())
    }

    /// Check `extend_with_nulls` on `TypedBuilder<Option<T>>`
    pub fn check_extend_with_nulls<T: ArrayElement>(
        backend_config: BackendConfig<Option<T>>,
        init_capacity: usize,
        num_nulls: usize,
    ) -> TestCaseResult
    where
        Option<T>: ArrayElement,
        BuilderBackend<Option<T>>: ValiditySlice,
    {
        let mut builder = TypedBuilder::<Option<T>>::with_config(BuilderConfig {
            capacity: Some(init_capacity),
            backend: backend_config,
        });
        builder.extend_with_nulls(num_nulls);
        check_extend_outcome(&builder, init_capacity, num_nulls)?;
        check_validity(&builder, &vec![false; num_nulls])?;
        Ok(())
    }
}
