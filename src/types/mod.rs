//! Rust equivalents of Arrow types

use crate::{impl_option_element, ArrayElement};
use arrow_array::builder::BooleanBuilder;

pub mod list;
pub mod primitive;

// SAFETY: By construction, it is enforced that Slice is &[Self]
unsafe impl ArrayElement for bool {
    type BuilderBackend = BooleanBuilder;
    type Value<'a> = Self;
    type Slice<'a> = &'a [Self];
    type ExtendFromSliceResult = ();
}
impl_option_element!(bool);
