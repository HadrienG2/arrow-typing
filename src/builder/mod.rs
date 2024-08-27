//! Strongly typed array builder

pub(crate) mod backend;
mod config;

use self::backend::{Backend, Capacity, Items, TypedBackend};
#[cfg(doc)]
use crate::element::option::OptionSlice;
use crate::element::{
    list::ListLike,
    option::{NullableElement, OptionalElement},
    ArrayElement,
};
use arrow_array::builder::ArrayBuilder;
use std::fmt::Debug;

/// Builder for an array whose elements are of type `T`
#[derive(Debug)]
#[repr(transparent)]
pub struct TypedBuilder<T: ArrayElement>(BuilderBackend<T>);
//
/// The following constructors are only available for simple element types like
/// primitive types which require no extra configuration. More complex element
/// types (e.g. fixed-sized lists of dynamically defined extent) may need to be
/// configured using the [`TypedBuilder::with_config()`] constructor.
impl<T: ArrayElement> TypedBuilder<T>
where
    BackendExtraConfig<T>: Default,
{
    /// Create a new array builder with the default configuration
    ///
    /// ```rust
    /// # use arrow_typing::TypedBuilder;
    /// let builder = TypedBuilder::<bool>::new();
    /// ```
    pub fn new() -> Self {
        Self(BuilderBackend::<T>::new(BuilderConfig::<T>::new()))
    }

    /// Create a new array builder with space for at least `capacity` elements
    ///
    /// ```rust
    /// # use arrow_typing::TypedBuilder;
    /// let builder = TypedBuilder::<bool>::with_capacity(42);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self(BuilderBackend::<T>::new(BuilderConfig::<T>::with_capacity(
            capacity,
        )))
    }
}
//
impl<T: ArrayElement> TypedBuilder<T> {
    /// Create a new array builder with a custom configuration
    ///
    /// See the documentation of [`BuilderConfig`] for more information and
    /// usage examples.
    pub fn with_config(config: BuilderConfig<T>) -> Self {
        Self(BuilderBackend::<T>::new(config))
    }

    /// Number of elements the array can hold without reallocating
    ///
    /// This operation is currently only available on `TypedBuilder`s of `Null`,
    /// bool and primitive types, as well as `Option`s and tuples of such types.
    ///
    /// It is conceptually similar to [`BuilderConfig::capacity()`], and
    /// everything that is explained in the documentation of this method
    /// concerning capacities of arrays of nontrivial types is also valid here.
    ///
    /// However, where the configuration's `capacity()` method recalls the
    /// storage capacity that you requested, this method lets you probe the
    /// storage capacity that gets actually allocated by the implementation at
    /// builder construction time, which may be higher.
    ///
    /// ```rust
    /// # use arrow_typing::TypedBuilder;
    /// let builder = TypedBuilder::<bool>::with_capacity(42);
    /// assert!(builder.capacity() >= 42);
    /// ```
    //
    // FIXME: Check if I could make arrow expose capacity() for the remaining
    //        builders that don't have it, e.g. GenericListBuilder, so that this
    //        can become a mandatory operation.
    pub fn capacity(&self) -> usize
    where
        BuilderBackend<T>: Capacity,
    {
        self.0.capacity()
    }

    /// Access the items builder from a list builder
    ///
    /// This operation is only available on array builders with [list-like
    /// elements](ListLike). It gives you read-only access to the inner item
    /// builder, which is used to hold the concatenated items from all
    /// previously inserted lists. This way, you can use specialized readout
    /// operations that are only available for your specific list item type.
    //
    // TODO: Add an example once I have a backend with an interesting read-only
    //       accessor. Or if all else fails, just use capacity().
    ///
    /// If you want to write into the inner items builder, use
    /// [`start_pushing()`](Self::start_pushing).
    pub fn items(&self) -> &TypedBuilder<T::Item>
    where
        T: ListLike,
        // TODO: Remove once implied by ListLike
        BuilderBackend<T>: Items<T>,
    {
        // SAFETY: This is safe because TypedBuilder is a repr(transparent)
        //         newtype of the underlying BuilderBackend.
        unsafe {
            std::mem::transmute::<&BuilderBackend<T::Item>, &TypedBuilder<T::Item>>(self.0.items())
        }
    }

    /// Append a single value into the builder
    ///
    /// For primitive types, this works just like [`Vec::push()`]. You give it a
    /// value of the type and it appends it at the end of the array.
    ///
    /// ```rust
    /// # use arrow_typing::TypedBuilder;
    /// let mut builder = TypedBuilder::<u8>::new();
    /// builder.push(123);
    /// ```
    ///
    /// For non-primitive types, however, [the value type accepted by this
    /// method](ArrayElement::WriteValue) can be a little less obvious. For
    /// example, to push a new list into a `TypedBuilder<List<P>>` where P is a
    /// primitive type, you would pass in a `&[P]` slice of that primitive type:
    ///
    /// ```rust
    /// # use arrow_typing::{TypedBuilder, element::list::List};
    /// let mut builder = TypedBuilder::<List<f32>>::new();
    /// builder.push(&[1.2, 3.4, 5.6]);
    /// ```
    ///
    /// As far as performance is concerned, note that for types with a more
    /// complex internal structure, like options or tuples, such element-wise
    /// insertion may be inefficient. Therefore, if you intend to insert many
    /// values, it is advised that you do not do so by calling this method in a
    /// loop, but instead look into the bulk insertion methods provided below.
    #[inline]
    pub fn push(&mut self, value: T::WriteValue<'_>) {
        self.0.push(value)
    }

    /// Start appending structured values via direct sub-builder access
    ///
    /// This operation is available when building arrays whose elements are
    /// composed of multiple sub-elements, as is the case for lists or tuples.
    /// It gives you write access to the inner [`TypedBuilder`]s used to insert
    /// these sub-elements, and thus lets you leverage specialized insertion
    /// methods that are only available for the specific sub-element type(s)
    /// that you are dealing with.
    ///
    /// For example, given a tuple array builder `TypedBuilder<(T, U)>`, this
    /// method lets you access the inner tuple of field builders `(&mut
    /// TypedBuilder<T>, &mut TypedBuilder<U>)`, for the purpose of inserting
    /// inner `(T, U)` tuples using the most efficient specialized insertion
    /// methods available for the concrete field types `T` and `U` that you are
    /// dealing with.
    //
    // TODO: Add a code example once I have struct builders
    ///
    /// In exchange for the extra performance optimization power that direct
    /// sub-builder access provides, you become responsible for maintaining the
    /// integrity constraints of the overarching structured array type. For
    /// example, a `TypedBuilder<(T, U)>` operates under the integrity
    /// constraint that the inner `TypedBuilder<T>` and `TypedBuilder<U>` field
    /// builders must contain the same number of elements. If your attempt to
    /// use `start_pushing()` in a manner which breaks this constraint, it will
    /// lead to a panic at the end of the transaction.
    ///
    /// See the documentation of [`Pusher`] for more information about how the
    /// object returned by this method should be used.
    pub fn start_pushing(&mut self) -> Pusher<T>
    where
        // TODO: Add StructuredElement trait to crate::element and make it a
        //       bound of the ListLike trait.
        T: StructuredElement,
    {
        Pusher::new(&mut self.0)
    }

    /// Efficiently append multiple values into the builder
    ///
    /// ```rust
    /// # use arrow_typing::TypedBuilder;
    /// let mut builder = TypedBuilder::<u32>::new();
    /// builder.extend_from_slice(&[
    ///     0xbaaaaaad,
    ///     0xcafed00d,
    ///     0xdead2bad,
    ///     0xfacefeed,
    ///     0xf0cacc1a,
    /// ]);
    /// ```
    ///
    /// For simple types, `T::WriteSlice` is just `&[T]`. But for efficiency
    /// reasons, slices of more complex types will have a less obvious columnar
    /// layout containing multiple inner Rust slices. For example, slices of
    /// options are passed as [`OptionSlice`]s:
    ///
    /// ```rust
    /// # use arrow_typing::{TypedBuilder, element::option::OptionSlice};
    /// let mut builder = TypedBuilder::<Option<f32>>::new();
    /// builder.extend_from_slice(OptionSlice {
    ///     values: &[
    ///         1.2,
    ///         3.4,
    ///         5.6,
    ///     ],
    ///     is_valid: &[
    ///         true,
    ///         false,
    ///         true
    ///     ]
    /// })?;
    /// # Ok::<_, anyhow::Error>(())
    /// ```
    ///
    /// While extending from a simple Rust slice always succeeds, extending from
    /// composite slice types like `OptionSlice` may fail if the inner subslices
    /// have differing lengths. Accordingly, this method returns `()` when
    /// `T::WriteSlice` is a simple Rust slice type, but `Result<(),
    /// ArrowError>` when `T::WriteSlice` is a composite slice type.
    //
    // FIXME: Add an example with structs once available
    pub fn extend_from_slice(&mut self, s: T::WriteSlice<'_>) -> T::ExtendFromSliceResult {
        self.0.extend_from_slice(s)
    }
}
//
impl<T: OptionalElement> TypedBuilder<Option<T>>
where
    // TODO: Remove bound once it becomes redundant with OptionalElement
    Option<T>: ArrayElement<BuilderBackend = BuilderBackend<T>>,
{
    /// Efficiently append multiple non-null values into the builder
    ///
    /// This operation is available for almost every `TypedBuilder` of
    /// `Option<T>` where `T` is an [`ArrayElement`], except for `Option<Null>`
    /// which does not exist in the Arrow data model.
    ///
    /// Given a slice of `T`, it lets you do the optimized equivalent of calling
    /// `push(Some(value))` in a loop for each value.
    ///
    /// ```rust
    /// # use arrow_typing::TypedBuilder;
    /// let mut builder = TypedBuilder::<Option<f64>>::new();
    /// builder.extend_from_value_slice(&[
    ///     2.4,
    ///     6.8,
    ///     10.12,
    /// ]);
    /// ```
    pub fn extend_from_value_slice(&mut self, vs: T::WriteSlice<'_>) -> T::ExtendFromSliceResult {
        self.0.extend_from_slice(vs)
    }
}
//
impl<T: ArrayElement> TypedBuilder<T> {
    /// Efficiently append multiple null values into the builder
    ///
    /// This operation is available when T is a [nullable
    /// type](NullableElement), i.e. `Null` or `Option<T>`.
    ///
    /// ```rust
    /// # use arrow_typing::{TypedBuilder, element::primitive::Null};
    /// let mut builder = TypedBuilder::<Null>::new();
    /// builder.extend_with_nulls(666);
    /// ```
    pub fn extend_with_nulls(&mut self, n: usize)
    where
        T: NullableElement,
    {
        self.0.extend_with_nulls(n)
    }

    /// Number of elements that were appended into this builder
    ///
    /// ```rust
    /// # use arrow_typing::TypedBuilder;
    /// let mut builder = TypedBuilder::<bool>::new();
    /// assert_eq!(builder.len(), 0);
    /// builder.push(true);
    /// assert_eq!(builder.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Truth that no elements were appended into this builder
    ///
    /// ```rust
    /// # use arrow_typing::TypedBuilder;
    /// let mut builder = TypedBuilder::<i32>::new();
    /// assert!(builder.is_empty());
    /// builder.push(42);
    /// assert!(!builder.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    // FIXME: Some equivalent of ArrayBuilder::finish() and finish_cloned that
    //        returns a TypedArrayRef

    /// Access the array elements that were inserted so far as a slice
    ///
    /// ```rust
    /// # use arrow_typing::{TypedBuilder, element::option::OptionSlice};
    /// let mut builder = TypedBuilder::<Option<f32>>::new();
    /// builder.extend_from_slice(OptionSlice {
    ///     values: &[1.2, 3.4, 5.6],
    ///     is_valid: &[true, false, true],
    /// })?;
    /// assert_eq!(
    ///     builder.as_slice(),
    ///     [Some(1.2), None, Some(5.6)].as_slice(),
    /// );
    /// # Ok::<_, anyhow::Error>(())
    /// ```
    pub fn as_slice(&self) -> T::ReadSlice<'_> {
        self.0.as_slice()
    }
}
//
impl<T: ArrayElement> Default for TypedBuilder<T>
where
    BackendExtraConfig<T>: Default,
{
    fn default() -> Self {
        Self::new()
    }
}
//
impl<'a, T: ArrayElement> Extend<T::WriteValue<'a>> for TypedBuilder<T> {
    fn extend<I: IntoIterator<Item = T::WriteValue<'a>>>(&mut self, iter: I) {
        for item in iter {
            self.push(item)
        }
    }
}

// Re-export builder configuration
pub use config::BuilderConfig;

/// Shortcut to the arrow builder type used to construct an array of Ts
type BuilderBackend<T> = <T as ArrayElement>::BuilderBackend;

/// Shortcut to the extra configuration parameters besides capacity
type BackendExtraConfig<T> = <BuilderBackend<T> as TypedBackend<T>>::ExtraConfig;

/// Shortcut to the alternate configuration methods besides new/with_capacity
type BackendAlternateConfig<T> = <BuilderBackend<T> as TypedBackend<T>>::AlternateConfig;

#[allow(private_bounds)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{option::OptionWriteSlice, Slice};
    use arrow_schema::ArrowError;
    use proptest::{prelude::*, sample::SizeRange, test_runner::TestCaseResult};

    /// Check outcome of initializing a `TypedBuilder` with some capacity
    pub fn check_init_with_capacity_outcome<T: ArrayElement>(
        builder: &TypedBuilder<T>,
        init_capacity: Option<usize>,
    ) -> TestCaseResult {
        if let (Some(init_capacity), Some(builder_capacity)) =
            (init_capacity, builder.0.capacity_opt())
        {
            prop_assert!(builder_capacity >= init_capacity);
        }
        prop_assert_eq!(builder.len(), 0);
        prop_assert!(builder.is_empty());

        let slice = builder.as_slice();
        prop_assert!(slice.is_consistent());
        prop_assert!(slice.is_empty());

        // FIXME: Build and check final array once possible
        Ok(())
    }

    /// Like `check_init_with_capacity`, but for both `T` and `Option<T>`
    ///
    /// For almost every [`ArrayElement`] type `T` with the exception of `Null`,
    /// `Option<T>` is also an `ArrayElement` and this test can be run.
    pub fn check_init_with_capacity_optional<T: ArrayElement>(
        make_backend_config: impl Fn() -> BackendExtraConfig<T>,
        capacity: usize,
    ) -> TestCaseResult
    where
        Option<T>: ArrayElement<BuilderBackend = BuilderBackend<T>>,
        BuilderBackend<T>: TypedBackend<Option<T>, ExtraConfig = BackendExtraConfig<T>>,
    {
        check_init_with_capacity_outcome(
            &TypedBuilder::<T>::with_config(BuilderConfig::Standard {
                capacity: Some(capacity),
                extra: make_backend_config(),
            }),
            Some(capacity),
        )?;
        check_init_with_capacity_outcome(
            &TypedBuilder::<Option<T>>::with_config(BuilderConfig::Standard {
                capacity: Some(capacity),
                extra: make_backend_config(),
            }),
            Some(capacity),
        )?;
        Ok(())
    }

    /// Check outcome of initializing a `TypedBuilder` with the default capacity
    pub fn check_init_default<T: ArrayElement>() -> TestCaseResult
    where
        BackendExtraConfig<T>: Default,
    {
        let mut builder = TypedBuilder::<T>::new();
        check_init_with_capacity_outcome(&builder, builder.0.capacity_opt())?;
        builder = TypedBuilder::<T>::default();
        check_init_with_capacity_outcome(&builder, builder.0.capacity_opt())?;
        builder = TypedBuilder::<T>::with_config(BuilderConfig::default());
        check_init_with_capacity_outcome(&builder, builder.0.capacity_opt())?;
        Ok(())
    }

    /// Like `check_init_default`, but for both `T` and `Option<T>`
    ///
    /// For almost every [`ArrayElement`] type `T` with the exception of `Null`,
    /// `Option<T>` is also an `ArrayElement` and this test can be run.
    pub fn check_init_default_optional<T: OptionalElement>() -> TestCaseResult
    where
        BackendExtraConfig<T>: Default,
        // TODO: Remove bounds they become redundant with OptionalElement
        Option<T>: ArrayElement,
        BackendExtraConfig<Option<T>>: Default,
    {
        check_init_default::<T>()?;
        check_init_default::<Option<T>>()?;
        Ok(())
    }

    /// Check outcome of inserting N values into a newly created TypedBuilder
    pub fn check_extend_outcome<'values, T: ArrayElement>(
        builder: &TypedBuilder<T>,
        init_capacity: usize,
        values: impl Iterator<Item = T::WriteValue<'values>> + Clone,
        mut values_eq: impl FnMut(T::ReadValue<'_>, T::WriteValue<'_>) -> bool,
    ) -> TestCaseResult {
        let num_elements = values.clone().count();
        if let Some(capacity) = builder.0.capacity_opt() {
            prop_assert!(capacity >= init_capacity.max(num_elements));
        }
        prop_assert_eq!(builder.len(), num_elements);
        prop_assert_eq!(builder.is_empty(), num_elements == 0);

        let slice = builder.as_slice();
        prop_assert!(slice.is_consistent());
        prop_assert_eq!(slice.len(), num_elements);
        for (read, written) in slice.iter_cloned().zip(values) {
            prop_assert!(values_eq(read, written));
        }

        // FIXME: Build and check final array once possible
        Ok(())
    }

    /// Turn the values_eq parameter of check_extend_outcome into a comparison
    /// of optional values
    pub fn options_eq<T: OptionalElement>(
        mut values_eq: impl FnMut(T::ReadValue<'_>, T::WriteValue<'_>) -> bool,
    ) -> impl FnMut(
        <Option<T> as ArrayElement>::ReadValue<'_>,
        <Option<T> as ArrayElement>::WriteValue<'_>,
    ) -> bool
    where
        // TODO: Remove these bounds once OptionalElement makes them implicit
        Option<T>: ArrayElement,
        for<'a> <Option<T> as ArrayElement>::ReadValue<'a>: Into<Option<T::ReadValue<'a>>>,
        for<'a> <Option<T> as ArrayElement>::WriteValue<'a>: Into<Option<T::WriteValue<'a>>>,
    {
        move |opt_read, opt_written| match (opt_read.into(), opt_written.into()) {
            (Some(read), Some(written)) => values_eq(read, written),
            (None, None) => true,
            (Some(_), None) | (None, Some(_)) => false,
        }
    }

    /// Check outcome of pushing a value into a newly created TypedBuilder
    pub fn check_push<T: ArrayElement>(
        config: BuilderConfig<T>,
        value: T::WriteValue<'_>,
        values_eq: impl FnMut(T::ReadValue<'_>, T::WriteValue<'_>) -> bool,
    ) -> TestCaseResult {
        let init_capacity = config.capacity();
        let mut builder = TypedBuilder::<T>::with_config(config);
        builder.push(value);
        check_extend_outcome(&builder, init_capacity, std::iter::once(value), values_eq)?;
        Ok(())
    }

    /// Check outcome of extending a builder of T or Option<T> with a slice of
    /// values
    pub fn check_extend_from_values<T: OptionalElement>(
        make_config: impl Fn() -> BuilderConfig<T>,
        values: T::WriteSlice<'_>,
        mut values_eq: impl FnMut(T::ReadValue<'_>, T::WriteValue<'_>) -> bool,
    ) -> TestCaseResult
    where
        // TODO: Remove bound once it becomes redundant with OptionalElement
        Option<T>: ArrayElement<BuilderBackend = BuilderBackend<T>>,
        BuilderBackend<T>: TypedBackend<
            Option<T>,
            ExtraConfig = BackendExtraConfig<T>,
            AlternateConfig = BackendAlternateConfig<T>,
        >,
        for<'a> T::WriteValue<'a>: Clone + Into<<Option<T> as ArrayElement>::WriteValue<'a>>,
        for<'a> <Option<T> as ArrayElement>::ReadValue<'a>: Into<Option<T::ReadValue<'a>>>,
        for<'a> <Option<T> as ArrayElement>::WriteValue<'a>: Into<Option<T::WriteValue<'a>>>,
    {
        let init_capacity = make_config().capacity();
        let make_value_builder = || TypedBuilder::<T>::with_config(make_config());
        {
            let mut value_builder = make_value_builder();
            value_builder.extend_from_slice(values);
            check_extend_outcome(
                &value_builder,
                init_capacity,
                values.iter_cloned(),
                &mut values_eq,
            )?;
        }
        {
            let mut value_builder = make_value_builder();
            value_builder.extend(values.iter_cloned());
            check_extend_outcome(
                &value_builder,
                init_capacity,
                values.iter_cloned(),
                &mut values_eq,
            )?;
        }

        let make_opt_builder =
            || TypedBuilder::<Option<T>>::with_config(make_config().cast::<Option<T>>());
        let mut options_eq = options_eq(values_eq);
        {
            let mut opt_builder = make_opt_builder();
            opt_builder.extend_from_value_slice(values);
            check_extend_outcome(
                &opt_builder,
                init_capacity,
                values.iter_cloned().map(Into::into),
                &mut options_eq,
            )?;
        }
        {
            let mut opt_builder = make_opt_builder();
            opt_builder.extend(values.iter_cloned().map(Into::into));
            check_extend_outcome(
                &opt_builder,
                init_capacity,
                values.iter_cloned().map(Into::into),
                &mut options_eq,
            )?;
        }
        Ok(())
    }

    /// Generate building blocks for an `OptionSlice<T>`
    pub fn option_vec<T: ArrayElement + Arbitrary>() -> impl Strategy<Value = (Vec<T>, Vec<bool>)> {
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
    pub fn option_vec_custom<T: ArrayElement, S: Strategy<Value = T>>(
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
    pub fn check_extend_from_options<T: OptionalElement>(
        config: BuilderConfig<Option<T>>,
        slice: OptionWriteSlice<T>,
        values_eq: impl FnMut(T::ReadValue<'_>, T::WriteValue<'_>) -> bool,
    ) -> TestCaseResult
    where
        // TODO: Remove bound once it becomes redundant with OptionalElement
        Option<T>: ArrayElement<ExtendFromSliceResult = Result<(), ArrowError>>,
        for<'a> OptionWriteSlice<'a, T>: Into<<Option<T> as ArrayElement>::WriteSlice<'a>>,
        for<'a> Option<T::WriteValue<'a>>: Into<<Option<T> as ArrayElement>::WriteValue<'a>>,
        for<'a> <Option<T> as ArrayElement>::ReadValue<'a>: Into<Option<T::ReadValue<'a>>>,
        for<'a> <Option<T> as ArrayElement>::WriteValue<'a>: Into<Option<T::WriteValue<'a>>>,
    {
        let init_capacity = config.capacity();
        let mut builder = TypedBuilder::<Option<T>>::with_config(config);
        let result = builder.extend_from_slice(slice.into());

        if slice.values.len() != slice.is_valid.len() {
            prop_assert!(result.is_err());
            check_init_with_capacity_outcome(&builder, Some(init_capacity))?;
            return Ok(());
        }

        prop_assert!(result.is_ok());
        check_extend_outcome(
            &builder,
            init_capacity,
            slice.iter_cloned().map(Into::into),
            options_eq(values_eq),
        )?;
        Ok(())
    }

    /// Check `extend_with_nulls` on `TypedBuilder<Option<T>>`
    pub fn check_extend_with_nulls<T: OptionalElement>(
        config: BuilderConfig<Option<T>>,
        num_nulls: usize,
        values_eq: impl FnMut(T::ReadValue<'_>, T::WriteValue<'_>) -> bool,
    ) -> TestCaseResult
    where
        // TODO: Remove bound once it becomes redundant with OptionalElement
        Option<T>: ArrayElement,
        for<'a> Option<T::WriteValue<'a>>: Into<<Option<T> as ArrayElement>::WriteValue<'a>>,
        for<'a> <Option<T> as ArrayElement>::ReadValue<'a>: Into<Option<T::ReadValue<'a>>>,
        for<'a> <Option<T> as ArrayElement>::WriteValue<'a>: Into<Option<T::WriteValue<'a>>>,
    {
        let init_capacity = config.capacity();
        let mut builder = TypedBuilder::<Option<T>>::with_config(config);
        builder.extend_with_nulls(num_nulls);
        check_extend_outcome(
            &builder,
            init_capacity,
            std::iter::repeat(None).take(num_nulls).map(Into::into),
            options_eq(values_eq),
        )?;
        Ok(())
    }
}
