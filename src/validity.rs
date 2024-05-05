//! Array validity bitmaps

use std::{
    cmp::Ordering,
    iter::{FusedIterator, Take},
};

/// Array validity bitmap
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ValiditySlice<'array> {
    /// Validity bitmap
    bitmap: &'array [u8],

    /// Number of trailing bits that have no associated array element
    ///
    /// Guaranteed to be in `0..=7`, will be 0 when `bitmap` is empty.
    trailer_len: u8,
}
//
impl<'array> ValiditySlice<'array> {
    /// Decode a validity slice from `arrow-rs`
    ///
    /// # Panics
    ///
    /// Panics if `array_len` is not in the expected `(bitmap.len() - 1) * 8..
    /// bitmap.len() * 8 range`.
    pub(crate) fn new(bitmap: &'array [u8], array_len: usize) -> Self {
        let error = "bitmap and array length don't match";
        let trailer_len = (bitmap.len() * 8).checked_sub(array_len).expect(error);
        assert!(trailer_len < 8, "{error}");
        Self {
            bitmap,
            trailer_len: trailer_len as u8,
        }
    }

    /// Number of elements in the validity bitmap
    pub const fn len(&self) -> usize {
        self.bitmap.len() * 8 - self.trailer_len as usize
    }

    /// Returns `true` if the source array contains no element.
    pub const fn is_empty(&self) -> bool {
        self.bitmap.is_empty()
    }

    /// Value of the `index`-th validity bit, if in bounds
    #[inline]
    pub fn get(&self, index: usize) -> Option<bool> {
        (index < self.len()).then(|| unsafe { self.get_unchecked(index) })
    }

    /// Value of the `index`-th validity bit, without bounds checking
    ///
    /// For a safe alternative see [`get`](Self::get).
    ///
    /// # Safety
    ///
    /// `index` must be in bounds or undefined behavior will ensue.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> bool {
        self.bitmap.get_unchecked(index / 8) & (1 << (index % 8)) != 0
    }

    /// Value of the `index`-th, with panic-based bounds checking
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn at(&self, index: usize) -> bool {
        self.get(index).expect("index is out of bounds")
    }

    /// Iterate over the slice
    pub fn iter(&self) -> Iter<'_> {
        let mut bytes = self.bitmap.iter();
        let current_byte = bytes.next().copied();
        (BitmapIter {
            bytes,
            current_byte,
            bit: 1,
        })
        .take(self.len())
    }
}
//
impl<'slice> IntoIterator for &'slice ValiditySlice<'slice> {
    type Item = bool;
    type IntoIter = Iter<'slice>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
//
impl PartialEq<&[bool]> for ValiditySlice<'_> {
    fn eq(&self, other: &&[bool]) -> bool {
        self.iter().eq(other.iter().copied())
    }
}
//
impl PartialOrd<&[bool]> for ValiditySlice<'_> {
    fn partial_cmp(&self, other: &&[bool]) -> Option<Ordering> {
        Some(self.iter().cmp(other.iter().copied()))
    }
}

/// Iterator over the elements of an array validity bitmap
pub type Iter<'slice> = Take<BitmapIter<'slice>>;

/// Iterator over the elements of an Arrow-style bitmap
#[derive(Clone, Debug, Default)]
pub struct BitmapIter<'bytes> {
    /// Iterator over the bitmap's bytes
    bytes: std::slice::Iter<'bytes, u8>,

    /// Last byte obtained from the `slice` iterator
    current_byte: Option<u8>,

    /// Currently targeted bit within current_byte
    bit: u8,
}
//
impl FusedIterator for BitmapIter<'_> {}
//
impl<'bytes> Iterator for BitmapIter<'bytes> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<bool> {
        // Access the current bit
        let byte = &self.current_byte?;
        let result = (byte & self.bit) != 0;

        // Move to the next byte/bit
        self.bit = self.bit.wrapping_shl(1);
        if self.bit == 0 {
            self.current_byte = self.bytes.next().copied();
            self.bit = 1;
        }

        // Return the result
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Generate validity bitmap building blocks
    fn building_blocks() -> impl Strategy<Value = (Vec<u8>, usize)> {
        prop_oneof![
            any::<Vec<bool>>().prop_map(|bits| { bits_to_bitmap(&bits) }),
            any::<(Vec<u8>, usize)>()
        ]
    }

    /// Convert validity bits into validity bitmap building blocks
    fn bits_to_bitmap(bits: &[bool]) -> (Vec<u8>, usize) {
        let array_len = bits.len();
        let bytes = bits
            .chunks(8)
            .map(|chunk| {
                let mut byte = 0u8;
                for (idx, bit) in chunk.iter().enumerate() {
                    byte |= (*bit as u8) << idx;
                }
                byte
            })
            .collect::<Vec<_>>();
        (bytes, array_len)
    }

    proptest! {
        #[test]
        fn init((bitmap, array_len) in building_blocks()) {
            let res = std::panic::catch_unwind(|| ValiditySlice::new(&bitmap, array_len));

            if bitmap.len() != array_len.div_ceil(8) {
                prop_assert!(res.is_err());
                return Ok(());
            }
            prop_assert!(res.is_ok());
            let validity = res.unwrap();

            prop_assert_eq!(validity.len(), array_len);
            prop_assert_eq!(validity.is_empty(), array_len == 0);
            prop_assert_eq!(validity.iter().count(), array_len);
            for (idx, bit) in validity.iter().enumerate() {
                prop_assert_eq!(bit, bitmap[idx / 8] & (1 << (idx % 8)) != 0);
            }
        }
    }

    /// Generate a validity bitmap, its unpacked bits, and an index into it
    fn bitmap_bits_index() -> impl Strategy<Value = ((Vec<u8>, usize), Vec<bool>, usize)> {
        let bits = any::<Vec<bool>>();
        bits.prop_flat_map(|bits| {
            let bitmap = Just(bits_to_bitmap(&bits));
            let index = prop_oneof![0..=bits.len(), bits.len()..];
            (bitmap, Just(bits), index)
        })
    }

    proptest! {
        #[test]
        fn index(((bitmap, array_len), bits, index) in bitmap_bits_index()) {
            let in_bounds = index < bits.len();
            let validity = ValiditySlice::new(&bitmap, array_len);

            prop_assert_eq!(validity, &bits[..]);

            prop_assert_eq!(validity.get(index), bits.get(index).copied());
            let index_res = std::panic::catch_unwind(|| validity.at(index));
            prop_assert_eq!(index_res.ok(), bits.get(index).copied());
            if in_bounds {
                prop_assert_eq!(unsafe { validity.get_unchecked(index) }, bits[index]);
            }
        }
    }
}
