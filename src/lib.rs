//! A layer on top of [`arrow`](https://docs.rs/arrow) which enables arrow
//! arrays to be built and accessed using strongly typed Rust APIs.

pub mod bitmap;
mod builder;
pub mod element;

pub use builder::*;

// NOTE: I tried to make this blanket-impl'd for Option<T> where
//       T::BuilderBackend: TypedBackend<Option<T>>, but this caused
//       problems down the line where backends were not recognized
//       by the trait solver as implementing TypedBackend<Option<T>>
//       because Option<T> did not implement ArrayElement. Let's
//       keep this macrofied for now.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_option_element {
    ($t:ty) => {
        // SAFETY: Option is not a primitive type and is therefore not
        //         affected by the safety precondition of ArrayElement
        unsafe impl $crate::element::ArrayElement for Option<$t> {
            type BuilderBackend = <$t as $crate::element::ArrayElement>::BuilderBackend;
            type WriteValue<'a> = Option<<$t as $crate::element::ArrayElement>::WriteValue<'a>>;
            type ReadValue<'a> = Option<<$t as $crate::element::ArrayElement>::ReadValue<'a>>;
            type WriteSlice<'a> = $crate::element::OptionWriteSlice<'a, $t>;
            type ReadSlice<'a> = $crate::element::OptionReadSlice<'a, $t>;
            type ExtendFromSliceResult = Result<(), arrow_schema::ArrowError>;
        }
    };
}

/// Re-expose the Slice API as inherent methods of the type
///
/// This macro should be called within an impl block of the slice type with
/// suitable generic parameters.
///
/// `is_consistent()` is not re-exported by default because it is not
/// appropriate for non-composite slice types, but you can also re-export it by
/// adding a `is_consistent` argument to the macro.
#[doc(hidden)]
#[macro_export]
macro_rules! inherent_slice_methods {
    (
        is_consistent,
        element: $element:ty
        $( , iter_lifetime: $iter_lifetime1:lifetime $(+ $iter_lifetimeN:lifetime)* )?
    ) => {
        /// Truth that all inner slices are consistent with each other
        ///
        /// You should check this and handle the inconsistent case before
        /// calling any other slice method.
        #[inline]
        pub fn is_consistent(&self) -> bool {
            <Self as $crate::element::Slice>::is_consistent(self)
        }

        $crate::inherent_slice_methods!(
            element: $element
            $( , iter_lifetime: $iter_lifetime1 $( + $iter_lifetimeN )* )?
        );
    };
    (
        element: $element:ty
        $( , iter_lifetime: $iter_lifetime1:lifetime $(+ $iter_lifetimeN:lifetime)* )?
    ) => {
        /// Number of slice elements
        #[inline]
        pub fn len(&self) -> usize {
            <Self as $crate::element::Slice>::len(self)
        }

        /// Truth that this slice has no elements
        #[inline]
        pub fn is_empty(&self) -> bool {
            <Self as $crate::element::Slice>::is_empty(self)
        }

        /// Value of the first element of the slice, or `None` if the slice is empty
        #[inline]
        pub fn first(&self) -> Option<$element> {
            <Self as $crate::element::Slice>::first_cloned(self)
        }

        /// Value of the last element of the slice, or `None` if it is empty
        #[inline]
        pub fn last(&self) -> Option<$element> {
            <Self as $crate::element::Slice>::last_cloned(self)
        }

        /// Value of the `index`-th slice element, if in bounds
        #[inline]
        pub fn get(&self, index: usize) -> Option<$element> {
            <Self as $crate::element::Slice>::get_cloned(self, index)
        }

        /// Value of the `index`-th slice element, without bounds checking
        ///
        /// For a safe alternative see [`get`](Self::get).
        ///
        /// # Safety
        ///
        /// Callers must ensure that `index < self.len()`. Additionally, if this
        /// is a composite slice type with an `is_consistent()` method, callers
        /// must ensure that the slice is indeed consistent before trusting the
        /// output of `len()`.
        #[inline]
        pub unsafe fn get_unchecked(&self, index: usize) -> $element {
            unsafe { <Self as $crate::element::Slice>::get_cloned_unchecked(self, index) }
        }

        /// Value of the `index`-th slice element, with panic-based bounds
        /// checking
        ///
        /// # Panics
        ///
        /// Panics if `index` is out of bounds.
        #[inline]
        pub fn at(&self, index: usize) -> $element {
            <Self as $crate::element::Slice>::at(self, index)
        }

        /// Iterate over copies of the elements of this slice
        #[allow(clippy::needless_lifetimes)]
        pub fn iter<
            'a $( : $iter_lifetime1 $( + $iter_lifetimeN )* )?
        >(&'a self) -> impl Iterator<Item = $element> + 'a {
            <Self as $crate::element::Slice>::iter_cloned(self)
        }

        /// Split this slice into two subslices at `mid`
        ///
        /// # Panics
        ///
        /// Panics if the slice has less than `mid` elements.
        pub fn split_at(&self, mid: usize) -> (Self, Self) {
            <Self as $crate::element::Slice>::split_at(self, mid)
        }
    };
}

/// Shared test utilities
#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    /// Maximum array length/capacity used in unit tests
    pub const MAX_CAPACITY: usize = 256;

    /// Generate a capacity between 0 and MAX_CAPACITY
    pub fn length_or_capacity() -> impl Strategy<Value = usize> {
        0..=MAX_CAPACITY
    }
}
