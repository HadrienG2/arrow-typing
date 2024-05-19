//! Array builder configuration

#[cfg(doc)]
use super::TypedBuilder;
use super::{
    backend::{list::ListConfig, Capacity, NoAlternateConfig, TypedBackend},
    BackendAlternateConfig, BackendExtraConfig,
};
use crate::element::{list::ListLike, ArrayElement};
use std::fmt::{self, Debug, Formatter};

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
