//! A layer on top of [`arrow`](https://docs.rs/arrow) which enables arrow
//! arrays to be built and accessed using strongly typed Rust APIs.

mod builder;
pub mod elements;
pub mod validity;

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
        unsafe impl $crate::elements::ArrayElement for Option<$t> {
            type BuilderBackend = <$t as $crate::elements::ArrayElement>::BuilderBackend;
            type Value<'a> = Option<<$t as $crate::elements::ArrayElement>::Value<'a>>;
            type Slice<'a> = $crate::elements::OptionSlice<'a, $t>;
            type ExtendFromSliceResult = Result<(), arrow_schema::ArrowError>;
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
