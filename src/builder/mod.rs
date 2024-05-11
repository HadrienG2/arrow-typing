//! Strongly typed array builder

pub(crate) mod backend;

use std::fmt::{self, Debug, Formatter};

use self::backend::{list::ListConfig, Backend, Capacity, NoAlternateConfig, TypedBackend};
#[cfg(doc)]
use crate::element::{primitive::PrimitiveType, OptionSlice};
use crate::{
    bitmap::Bitmap,
    element::{list::ListLike, ArrayElement, NullableElement},
};
use arrow_array::builder::ArrayBuilder;

/// Builder for an array whose elements are of type `T`
#[derive(Debug)]
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
    pub fn capacity(&self) -> usize
    where
        BuilderBackend<T>: Capacity,
    {
        self.0.capacity()
    }

    /// Append a single value into the builder
    ///
    /// ```rust
    /// # use arrow_typing::TypedBuilder;
    /// let mut builder = TypedBuilder::<u8>::new();
    /// builder.push(123);
    /// ```
    //
    // FIXME: Example with a type where T::Value is less obvious, like List
    ///
    /// For types with a complex internal structure, such element-wise insertion
    /// may be inefficient. Therefore, if you intend to insert many values, it
    /// is advised that you do not do so by calling this method in a loop, but
    /// instead look into the bulk insertion methods below.
    #[inline]
    pub fn push(&mut self, value: T::WriteValue<'_>) {
        self.0.push(value)
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
    /// For simple types, `T::Slice` is just `&[T]`. But for efficiency reasons,
    /// slices of more complex types will have a less obvious columnar layout
    /// containing multiple inner Rust slices. For example, slices of options
    /// are passed as [`OptionSlice`]s:
    ///
    /// ```rust
    /// # use arrow_typing::{TypedBuilder, element::OptionSlice};
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
    /// `T::Slice` is a simple Rust slice type, but `Result<(), ArrowError>`
    /// when `T::Slice` is a composite slice type.
    //
    // FIXME: Add an example with structs once available
    pub fn extend_from_slice(&mut self, s: T::WriteSlice<'_>) -> T::ExtendFromSliceResult {
        self.0.extend_from_slice(s)
    }
}
//
impl<T: ArrayElement> TypedBuilder<Option<T>>
where
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
    ///
    /// It may return `None` when all elements are known to be valid. Otherwise,
    /// it will return a `&[bool]`-like [`Bitmap`] which can be used to check
    /// which elements are valid.
    ///
    /// ```rust
    /// # use arrow_typing::{TypedBuilder, element::OptionSlice};
    /// let mut builder = TypedBuilder::<Option<f32>>::new();
    /// let validity: &[bool] = &[true, false, true];
    /// builder.extend_from_slice(OptionSlice {
    ///     values: &[
    ///         1.2,
    ///         3.4,
    ///         5.6,
    ///     ],
    ///     is_valid: validity
    /// })?;
    /// assert_eq!(
    ///     builder.validity_slice().expect("not all elements are valid"),
    ///     validity
    /// );
    /// # Ok::<_, anyhow::Error>(())
    /// ```
    pub fn validity_slice(&self) -> Option<Bitmap<'_>> {
        use backend::ValiditySlice;
        self.0
            .validity_slice()
            .map(|bitmap| Bitmap::new(bitmap, self.len()))
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

/// Configuration needed to construct a [`TypedBuilder`]
pub enum BuilderConfig<T: ArrayElement> {
    /// Configuration for the standard new/with_capacity constructor
    #[doc(hidden)]
    Standard {
        /// Minimal number of elements this builder can accept without reallocating
        capacity: Option<usize>,

        /// Backend-specific configuration
        extra: BackendExtraConfig<T>,
    },

    /// Configuration for alternate constructors, if available
    #[doc(hidden)]
    Alternate(BackendAlternateConfig<T>),
}
//
/// The following constructors are available for simple array element types like
/// primitive types, where there is an obvious default builder configuration.
///
/// More complex element types that do not have an obvious default configuration
/// (e.g. fixed-sized lists of dynamically defined extent) will need to be
/// configured using one of the other constructors.
impl<T: ArrayElement> BuilderConfig<T>
where
    BackendExtraConfig<T>: Default,
{
    /// Configure a builder with its default configuration
    ///
    /// ```rust
    /// # use arrow_typing::{BuilderConfig, TypedBuilder};
    /// // The following two declarations are equivalent
    /// let builder1 = TypedBuilder::<f32>::new();
    /// let builder2 = TypedBuilder::<f32>::with_config(
    ///     BuilderConfig::new()
    /// );
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure a builder with space for at least `capacity` elements
    ///
    /// ```rust
    /// # use arrow_typing::{BuilderConfig, TypedBuilder};
    /// // The following two declarations are equivalent
    /// let builder1 = TypedBuilder::<u8>::with_capacity(123);
    /// let builder2 = TypedBuilder::<u8>::with_config(
    ///     BuilderConfig::with_capacity(123)
    /// );
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self::Standard {
            capacity: Some(capacity),
            extra: Default::default(),
        }
    }
}
//
/// The following methods are available on `BuilderConfig<List<T, _>>` and
/// `BuilderConfig<Option<List<T, _>>>`.
impl<List: ListLike> BuilderConfig<List>
where
    // TODO: Remove bound once ListLike trait can be made more specific
    List::BuilderBackend: TypedBackend<
        List,
        ExtraConfig = ListConfig<List::Item>,
        AlternateConfig = NoAlternateConfig,
    >,
{
    /// Configure a builder for an array of lists
    ///
    /// This method is meant be used as an alternative to
    /// `new()`/`with_capacity()` when either of the following is true:
    ///
    /// - The list item type does not have a default configuration (e.g. this is
    ///   a list of tuples or a list of fixed-size lists), and thus the easy
    ///   `new()` and `with_capacity()` constructors are unavailable.
    /// - You do not want to use the default configuration of the item type
    ///   (e.g. you want the inner array of items to have a certain capacity in
    ///   order to avoid reallocations when list items are pushed).
    ///
    /// The `capacity` argument can be used to set the list builder's capacity,
    /// i.e. the minimal number of lists that can be pushed without an offset
    /// buffer reallocation. When it is set to `None` this constructor behaves
    /// like `new()`, and when it is set to `Some(capacity)` this constructor
    /// behaves like `with_capacity()`.
    ///
    /// The `item_config` argument is used to configure the inner
    /// `TypedBuilder<List::Item>` on top of which the `TypedBuilder<List>` is
    /// built.
    ///
    /// ```rust
    /// # use arrow_typing::{BuilderConfig, TypedBuilder, element::list::List};
    /// // Configure an array of optional lists of booleans with storage for
    /// // 42 lists and a total of 666 boolean items across all lists.
    /// let item_config = BuilderConfig::with_capacity(666);
    /// let list_config = BuilderConfig::new_list(Some(42), item_config);
    /// let mut list_builder: TypedBuilder<Option<List<bool>>> =
    ///     TypedBuilder::with_config(list_config);
    /// ```
    pub fn new_list(capacity: Option<usize>, item_config: BuilderConfig<List::Item>) -> Self {
        Self::Standard {
            capacity,
            extra: ListConfig {
                item_name: None,
                item_config,
            },
        }
    }

    /// Set the name of the list array's item field
    ///
    /// By default, list items get the conventional field name "item".
    ///
    /// ```rust
    /// # use arrow_typing::{BuilderConfig, element::{list::List, primitive::Null}};
    /// let list_config: BuilderConfig<List<Null>> =
    ///     BuilderConfig::new().with_item_name("null_item");
    /// ```
    pub fn with_item_name(self, item_name: impl ToString) -> Self {
        let Self::Standard {
            capacity,
            mut extra,
        } = self
        else {
            unreachable!()
        };
        extra.item_name = Some(item_name.to_string());
        Self::Standard { capacity, extra }
    }
}
//
impl<T: ArrayElement> BuilderConfig<T> {
    /// Expected capacity of an array builder made using this configuration
    ///
    /// In the case of types that are internally stored as multiple columnar
    /// buffers, like tuples, a lower bound on the capacity of all underlying
    /// columns is returned.
    //
    // FIXME: Example once tuples available
    ///
    /// In the case of lists, capacity should be understood as the number of
    /// lists that can be pushed without reallocating _assuming enough capacity
    /// to store all items in the inner items builder_.
    ///
    /// ```rust
    /// # use arrow_typing::{TypedBuilder, BuilderConfig};
    /// #
    /// let requested_capacity = 987;
    /// let config = BuilderConfig::with_capacity(requested_capacity);
    /// assert_eq!(config.capacity(), requested_capacity);
    ///
    /// let builder = TypedBuilder::<i64>::with_config(config);
    /// assert!(builder.capacity() >= requested_capacity);
    /// ```
    pub fn capacity(&self) -> usize {
        match self {
            Self::Standard { capacity, .. } => capacity.unwrap_or(0),
            Self::Alternate(alt) => alt.capacity(),
        }
    }

    /// Cast between compatible configuration types
    ///
    /// Configuration types are compatible when they contain the same
    /// information. The following configuration types are compatible today and
    /// guaranteed to remain compatible in the future:
    ///
    /// - `BuilderConfig<T>` and `BuilderConfig<Option<T>>` for any array
    ///   element type `T` other than `Null`.
    ///
    /// Other configuration types may be "accidentally" compatible at present
    /// time, but are not guaranteed to remain compatible throughout future
    /// releases of `arrow-rs`. Therefore, do not rely on any configuration cast
    /// other than the aforementioned ones.
    ///
    /// ```rust
    /// # use arrow_typing::BuilderConfig;
    /// let value_config: BuilderConfig<bool> = BuilderConfig::new();
    /// let option_config: BuilderConfig<Option<bool>> = value_config.cast();
    /// ```
    pub fn cast<U: ArrayElement>(self) -> BuilderConfig<U>
    where
        U::BuilderBackend: TypedBackend<
            U,
            ExtraConfig = BackendExtraConfig<T>,
            AlternateConfig = BackendAlternateConfig<T>,
        >,
    {
        match self {
            Self::Standard { capacity, extra } => BuilderConfig::Standard { capacity, extra },
            Self::Alternate(alt) => BuilderConfig::Alternate(alt),
        }
    }
}
//
impl<T: ArrayElement> Clone for BuilderConfig<T>
where
    BackendExtraConfig<T>: Clone,
    BackendAlternateConfig<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Standard { capacity, extra } => Self::Standard {
                capacity: *capacity,
                extra: extra.clone(),
            },
            Self::Alternate(alternate) => Self::Alternate(alternate.clone()),
        }
    }
}
//
impl<T: ArrayElement> Copy for BuilderConfig<T>
where
    BackendExtraConfig<T>: Copy,
    BackendAlternateConfig<T>: Copy,
{
}
//
impl<T: ArrayElement> Debug for BuilderConfig<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard { capacity, extra } => f
                .debug_struct("BuilderConfig::Standard")
                .field("capacity", &capacity)
                .field("extra", &extra)
                .finish(),
            Self::Alternate(alternate) => f
                .debug_tuple("BuilderConfig::Alternate")
                .field(&alternate)
                .finish(),
        }
    }
}
//
impl<T: ArrayElement> Default for BuilderConfig<T>
where
    BackendExtraConfig<T>: Default,
{
    fn default() -> Self {
        Self::Standard {
            capacity: None,
            extra: Default::default(),
        }
    }
}
//
impl<T: ArrayElement> PartialEq for BuilderConfig<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Standard {
                    capacity: c1,
                    extra: e1,
                },
                Self::Standard {
                    capacity: c2,
                    extra: e2,
                },
            ) => c1 == c2 && e1 == e2,
            (Self::Alternate(a1), Self::Alternate(a2)) => a1 == a2,
            (Self::Standard { .. }, Self::Alternate(_))
            | (Self::Alternate(_), Self::Standard { .. }) => false,
        }
    }
}

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
    use crate::element::{OptionWriteSlice, Slice};
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
    pub fn check_init_default_optional<T: ArrayElement>() -> TestCaseResult
    where
        Option<T>: ArrayElement,
        BackendExtraConfig<T>: Default,
        BackendExtraConfig<Option<T>>: Default,
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
        if let Some(capacity) = builder.0.capacity_opt() {
            prop_assert!(capacity >= init_capacity.max(num_elements));
        }
        prop_assert_eq!(builder.len(), num_elements);
        prop_assert_eq!(builder.is_empty(), num_elements == 0);
        // FIXME: Build and check final array once possible
        Ok(())
    }

    /// Check outcome of pushing a value into a newly created TypedBuilder
    pub fn check_push<T: ArrayElement>(
        config: BuilderConfig<T>,
        value: T::WriteValue<'_>,
    ) -> TestCaseResult {
        let init_capacity = config.capacity();
        let mut builder = TypedBuilder::<T>::with_config(config);
        builder.push(value);
        check_extend_outcome(&builder, init_capacity, 1)?;
        Ok(())
    }

    /// Like `check_push`, but with `Option<T>` and validity bitmap checking
    pub fn check_push_option<T: ArrayElement>(
        config: BuilderConfig<Option<T>>,
        value: Option<T>,
    ) -> TestCaseResult
    where
        Option<T>: ArrayElement,
        BuilderBackend<Option<T>>: ValiditySlice,
        for<'a> Option<T>: Into<<Option<T> as ArrayElement>::WriteValue<'a>>,
    {
        let init_capacity = config.capacity();
        let mut builder = TypedBuilder::<Option<T>>::with_config(config);
        let valid = value.is_some();
        builder.push(value.into());
        check_extend_outcome(&builder, init_capacity, 1)?;
        check_validity(&builder, &[valid])?;
        Ok(())
    }

    /// Check outcome of extending a builder of T or Option<T> with a slice of
    /// values
    pub fn check_extend_from_values<T: ArrayElement>(
        make_config: impl Fn() -> BuilderConfig<T>,
        values: T::WriteSlice<'_>,
    ) -> TestCaseResult
    where
        Option<T>: ArrayElement<BuilderBackend = BuilderBackend<T>>,
        BuilderBackend<T>: TypedBackend<
            Option<T>,
            ExtraConfig = BackendExtraConfig<T>,
            AlternateConfig = BackendAlternateConfig<T>,
        >,
        BuilderBackend<Option<T>>: ValiditySlice,
        for<'a> T::WriteValue<'a>: Clone + Into<<Option<T> as ArrayElement>::WriteValue<'a>>,
    {
        let init_capacity = make_config().capacity();
        let make_value_builder = || TypedBuilder::<T>::with_config(make_config());
        {
            let mut value_builder = make_value_builder();
            value_builder.extend_from_slice(values);
            check_extend_outcome(&value_builder, init_capacity, values.len())?;
        }
        {
            let mut value_builder = make_value_builder();
            value_builder.extend(values.iter_cloned());
            check_extend_outcome(&value_builder, init_capacity, values.len())?;
        }

        let make_opt_builder =
            || TypedBuilder::<Option<T>>::with_config(make_config().cast::<Option<T>>());
        {
            let mut opt_builder = make_opt_builder();
            opt_builder.extend_from_value_slice(values);
            check_extend_outcome(&opt_builder, init_capacity, values.len())?;
            check_validity(&opt_builder, &vec![true; values.len()])?;
        }
        {
            let mut opt_builder = make_opt_builder();
            opt_builder.extend(values.iter_cloned().map(Into::into));
            check_extend_outcome(&opt_builder, init_capacity, values.len())?;
            check_validity(&opt_builder, &vec![true; values.len()])?;
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
    pub fn check_extend_from_options<T: ArrayElement>(
        config: BuilderConfig<Option<T>>,
        slice: OptionWriteSlice<T>,
    ) -> TestCaseResult
    where
        Option<T>: ArrayElement<ExtendFromSliceResult = Result<(), ArrowError>>,
        for<'a> OptionWriteSlice<'a, T>: Into<<Option<T> as ArrayElement>::WriteSlice<'a>>,
        BuilderBackend<Option<T>>: ValiditySlice,
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
        check_extend_outcome(&builder, init_capacity, slice.values.len())?;
        check_validity(&builder, slice.is_valid)?;
        Ok(())
    }

    /// Check `extend_with_nulls` on `TypedBuilder<Option<T>>`
    pub fn check_extend_with_nulls<T: ArrayElement>(
        config: BuilderConfig<Option<T>>,
        num_nulls: usize,
    ) -> TestCaseResult
    where
        Option<T>: ArrayElement,
        BuilderBackend<Option<T>>: ValiditySlice,
    {
        let init_capacity = config.capacity();
        let mut builder = TypedBuilder::<Option<T>>::with_config(config);
        builder.extend_with_nulls(num_nulls);
        check_extend_outcome(&builder, init_capacity, num_nulls)?;
        check_validity(&builder, &vec![false; num_nulls])?;
        Ok(())
    }
}
