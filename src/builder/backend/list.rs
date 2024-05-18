//! Strong typing layer on top of [`GenericListBuilder`]

use super::{Backend, NoAlternateConfig, TypedBackend};
use crate::{
    bitmap::Bitmap,
    builder::BuilderConfig,
    element::{
        list::{List, ListWriteSlice, OptionListWriteSlice},
        ArrayElement, Slice,
    },
};
use arrow_array::{
    builder::{ArrayBuilder, GenericListBuilder},
    OffsetSizeTrait,
};
use arrow_schema::{ArrowError, DataType, Field, FieldRef};
use std::fmt::{self, Debug, Formatter};

impl<OffsetSize: OffsetSizeTrait, T: ArrayBuilder + Debug> Backend
    for GenericListBuilder<OffsetSize, T>
{
    #[cfg(test)]
    fn capacity_opt(&self) -> Option<usize> {
        None
    }

    fn extend_with_nulls(&mut self, n: usize) {
        for _ in 0..n {
            self.append_null()
        }
    }

    type ValiditySlice<'a> = Bitmap<'a>;

    fn validity_slice(&self) -> Option<Self::ValiditySlice<'_>> {
        self.validity_slice()
            .map(|validity| Bitmap::new(validity, self.len()))
    }
}

/// Extra configuration requested by the list backend
pub struct ListConfig<Item: ArrayElement> {
    /// Name of the list item field ("item" by default)
    pub item_name: Option<String>,

    /// Configuration of the inner values
    pub item_config: BuilderConfig<Item>,
}
//
impl<Item: ArrayElement> ListConfig<Item> {
    /// Get the appropriate `Field` to use for this list
    fn make_field(&self) -> Field {
        Item::BuilderBackend::make_field(
            &self.item_config,
            self.item_name
                .clone()
                .unwrap_or_else(|| String::from("item")),
        )
    }
}
//
impl<Item: ArrayElement> Clone for ListConfig<Item>
where
    BuilderConfig<Item>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            item_name: self.item_name.clone(),
            item_config: self.item_config.clone(),
        }
    }
}
//
impl<Item: ArrayElement> Debug for ListConfig<Item> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ListConfig")
            .field("item_name", &self.item_name)
            .field("item_config", &self.item_config)
            .finish()
    }
}
//
impl<Item: ArrayElement> Default for ListConfig<Item>
where
    BuilderConfig<Item>: Default,
{
    fn default() -> Self {
        Self {
            item_name: None,
            item_config: Default::default(),
        }
    }
}
//
impl<Item: ArrayElement> PartialEq for ListConfig<Item> {
    fn eq(&self, other: &Self) -> bool {
        self.item_name == other.item_name && self.item_config == other.item_config
    }
}

/// Extract the capacity and ListConfig<T> out of a BuilderConfig<List<T>>, a
/// BuilderConfig<Option<List<T>>>, or a reference to either, return values or
/// references as appropriate for the input type.
macro_rules! into_capacity_and_list_config {
    ($config:expr) => {{
        let BuilderConfig::Standard {
            capacity,
            extra: list_config,
        } = $config
        else {
            unreachable!()
        };
        (capacity, list_config)
    }};
}

// Common parts of TypedBackend impls for List<Item> and Option<List<Item>>
macro_rules! typed_backend_common {
    ($element_type:ty, $is_option:expr) => {
        type ExtraConfig = ListConfig<Item>;
        type AlternateConfig = NoAlternateConfig;

        fn make_field(config: &BuilderConfig<$element_type>, name: String) -> Field {
            let (_capacity, list_config) = into_capacity_and_list_config!(config);
            let field: FieldRef = list_config.make_field().into();
            let data_type = match std::mem::size_of::<OffsetSize>() {
                4 => DataType::List(field),
                8 => DataType::LargeList(field),
                _ => unreachable!(),
            };
            Field::new(name, data_type, $is_option)
        }

        fn new(config: BuilderConfig<$element_type>) -> Self {
            let (capacity, list_config) = into_capacity_and_list_config!(config);
            let field = list_config.make_field();
            let items_builder = Item::BuilderBackend::new(list_config.item_config);
            let list_builder = if let Some(capacity) = capacity {
                Self::with_capacity(items_builder, capacity)
            } else {
                Self::new(items_builder)
            };
            list_builder.with_field(field)
        }
    };
}

impl<OffsetSize: OffsetSizeTrait, Item: ArrayElement> TypedBackend<List<Item, OffsetSize>>
    for GenericListBuilder<OffsetSize, Item::BuilderBackend>
{
    typed_backend_common!(List<Item, OffsetSize>, false);

    #[inline]
    fn push(&mut self, s: Item::WriteSlice<'_>) {
        self.values().extend_from_slice(s);
        self.append(true)
    }

    fn extend_from_slice(&mut self, s: ListWriteSlice<'_, Item>) -> Result<(), ArrowError> {
        if !s.is_consistent() {
            return Err(ArrowError::InvalidArgumentError(
                "sum of sublist lengths should equate value buffer length".to_string(),
            ));
        }
        for sublist in s.iter_cloned() {
            <Self as TypedBackend<List<Item, OffsetSize>>>::push(self, sublist);
        }
        Ok(())
    }
}

impl<OffsetSize: OffsetSizeTrait, Item: ArrayElement> TypedBackend<Option<List<Item, OffsetSize>>>
    for GenericListBuilder<OffsetSize, Item::BuilderBackend>
{
    typed_backend_common!(Option<List<Item, OffsetSize>>, true);

    #[inline]
    fn push(&mut self, s: Option<Item::WriteSlice<'_>>) {
        if let Some(slice) = s {
            self.values().extend_from_slice(slice);
            self.append(true)
        } else {
            self.append(false)
        }
    }

    fn extend_from_slice(&mut self, s: OptionListWriteSlice<'_, Item>) -> Result<(), ArrowError> {
        if !s.is_consistent() {
            return Err(ArrowError::InvalidArgumentError(
                "sum of sublist lengths should equate value buffer length".to_string(),
            ));
        }
        for sublist in s.iter_cloned() {
            <Self as TypedBackend<Option<List<Item, OffsetSize>>>>::push(self, sublist);
        }
        Ok(())
    }
}

// FIXME: Add tests once full API is covered
