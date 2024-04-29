//! Mechanisms to build arrow arrays

use std::fmt::Debug;

use crate::ArrayElement;
use arrow_array::builder::{ArrayBuilder, BooleanBuilder, NullBuilder};

/// Common facade over type-safe builders for all data types
#[derive(Debug)]
pub struct TypedBuilder<T: ArrayElement>(<T as ArrayElement>::BuilderBackend);
//
impl<T: ArrayElement> TypedBuilder<T> {
    /// Create a new builder
    pub fn new(params: ConstructorParams<T>) -> Self {
        Self(T::BuilderBackend::new(params))
    }

    /// Create a new builder with space for `capacity` elements
    pub fn with_capacity(params: ConstructorParams<T>, capacity: usize) -> Self {
        Self(T::BuilderBackend::with_capacity(params, capacity))
    }

    /// Append a single value into the builder
    #[inline]
    pub fn push(&mut self, value: T::Value<'_>) {
        self.0.push(value)
    }

    /// Number of elements that were pushed into this builder
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Truth that no elements were pushed into this builder
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    // TODO: Some equivalent of ArrayBuilder::finish() that returns a typed
    //       ArrayRef
}
//
impl<T: ArrayElement> TypedBuilder<T>
where
    BuilderBackend<T>: AppendSlice<T>,
{
    /// Append a slice of elements into the builder
    pub fn append_slice(&mut self, s: &[T]) {
        self.0.append_slice(s)
    }
}
//
impl<T: ArrayElement> TypedBuilder<T>
where
    BuilderBackend<T>: AppendNulls,
{
    /// Efficiently push `n` `None`s into the builder
    pub fn append_nulls(&mut self, n: usize) {
        self.0.append_nulls(n)
    }
}
//
impl<T: ArrayElement> TypedBuilder<T>
where
    BuilderBackend<T>: Capacity,
{
    /// Number of elements the array can hold without reallocating
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }
}
//
impl<T: ArrayElement> Default for TypedBuilder<T>
where
    ConstructorParams<T>: Default,
{
    fn default() -> Self {
        Self(T::BuilderBackend::new(Default::default()))
    }
}

/// Shortcut to the arrow builder type used to construct an array of Ts
pub type BuilderBackend<T> = <T as ArrayElement>::BuilderBackend;

/// Shortcut to the constructor parameters needed to construct an array of Ts
pub type ConstructorParams<T> =
    <<T as ArrayElement>::BuilderBackend as Constructor>::ConstructorParams;

/// Arrow mechanism to build an arrow array of objects of type T
pub trait Backend<T: ArrayElement + ?Sized>: ArrayBuilder + Constructor + Debug {
    /// Append a single value into the builder
    fn push(&mut self, v: T::Value<'_>);
}
//
/// Mechanism to construct an array builder
pub trait Constructor {
    /// Constructor parameters other than inner array builders
    type ConstructorParams;

    /// Create a new builder with no underlying buffer allocation
    fn new(params: Self::ConstructorParams) -> Self;

    /// Create a new builder with space for `capacity` elements
    fn with_capacity(params: Self::ConstructorParams, capacity: usize) -> Self;
}
//
/// Optional mechanism to query the current capacity of an array builder
pub trait Capacity {
    /// Number of elements the array can hold without reallocating
    fn capacity(&self) -> usize;
}
//
/// Optional mechanism for bulk insertion of elements into an array builder
pub trait AppendSlice<T> {
    /// Append a slice of elements into the builder
    fn append_slice(&mut self, s: &[T]);
}
//
/// Optional mechanism for bulk insertion of nulls into an array builder
pub trait AppendNulls {
    /// Efficiently push `n` `None`s into the builder
    fn append_nulls(&mut self, n: usize);
}

// Allow NullBuilder as a builder backend for ()
impl Constructor for NullBuilder {
    type ConstructorParams = ();

    fn new(_params: ()) -> Self {
        Self::new()
    }

    fn with_capacity(_params: (), capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }
}
//
impl Capacity for NullBuilder {
    fn capacity(&self) -> usize {
        self.capacity()
    }
}
//
impl AppendNulls for NullBuilder {
    fn append_nulls(&mut self, n: usize) {
        self.append_nulls(n)
    }
}
//
impl Backend<()> for NullBuilder {
    #[inline]
    fn push(&mut self, _v: ()) {
        self.append_null()
    }
}

// Allow BooleanBuilder as a builder backend for bool and Option<bool>
impl Constructor for BooleanBuilder {
    type ConstructorParams = ();

    fn new(_params: ()) -> Self {
        Self::new()
    }

    fn with_capacity(_params: (), capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }
}
//
impl Capacity for BooleanBuilder {
    fn capacity(&self) -> usize {
        self.capacity()
    }
}
//
impl AppendSlice<bool> for BooleanBuilder {
    fn append_slice(&mut self, s: &[bool]) {
        self.append_slice(s)
    }
}
//
impl AppendNulls for BooleanBuilder {
    fn append_nulls(&mut self, n: usize) {
        self.append_nulls(n)
    }
}
//
impl Backend<bool> for BooleanBuilder {
    #[inline]
    fn push(&mut self, v: bool) {
        self.append_value(v)
    }
}
//
impl Backend<Option<bool>> for BooleanBuilder {
    #[inline]
    fn push(&mut self, v: Option<bool>) {
        self.append_option(v)
    }
}

// TODO: Add tests
