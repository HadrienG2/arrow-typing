//! Array validity bitmaps

use std::{
    cmp::Ordering,
    iter::{FusedIterator, Take},
};

use crate::element::Slice;

/// Array validity bitmap
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ValiditySlice<'array> {
    /// Validity bitmap
    bitmap: &'array [u8],

    /// Number of leading bits in the first byte of the bitmap that have no
    /// associated array element
    ///
    /// Guaranteed to be in `0..=7`, will be equal to `8 - trailer_len` when the
    /// bitmap is empty.
    header_len: u8,

    /// Number of trailing bits in the last byte of the bitmap that have no
    /// associated array element
    ///
    /// Guaranteed to be in `0..=7`, will be equal to `8 - header_len` when the
    /// bitmap is empty.
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
            header_len: 0,
            trailer_len: trailer_len as u8,
        }
    }

    crate::inherent_slice_methods!(element: bool, iter_lifetime: 'array);
}
//
impl<'slice> IntoIterator for &'slice ValiditySlice<'slice> {
    type Item = bool;
    type IntoIter = Iter<'slice>;
    fn into_iter(self) -> Self::IntoIter {
        let mut bytes = self.bitmap.iter();
        let current_byte = bytes.next().copied();
        (BitmapIter {
            bytes,
            current_byte,
            bit: 1 << self.header_len,
        })
        .take(self.len())
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
//
impl Slice for ValiditySlice<'_> {
    type Value = bool;

    fn has_consistent_lens(&self) -> bool {
        true
    }

    #[inline]
    fn len(&self) -> usize {
        self.bitmap.len() * 8 - (self.header_len + self.trailer_len) as usize
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, index: usize) -> bool {
        let bit = index + self.header_len as usize;
        self.bitmap.get_unchecked(bit / 8) & (1 << (bit % 8)) != 0
    }

    fn iter_cloned(&self) -> impl Iterator<Item = bool> + '_ {
        let mut bytes = self.bitmap.iter();
        let current_byte = bytes.next().copied();
        (BitmapIter {
            bytes,
            current_byte,
            bit: 1 << self.header_len,
        })
        .take(self.len())
    }

    fn split_at(&self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.len(), "split point is out of bounds");

        let mid_bit = mid + self.header_len as usize;
        let num_head_bytes = mid_bit.div_ceil(8);
        let head_bitmap = &self.bitmap[..num_head_bytes];
        let head = Self {
            bitmap: head_bitmap,
            header_len: self.header_len,
            trailer_len: (head_bitmap.len() * 8 - mid_bit) as u8,
        };
        debug_assert_eq!(head.len(), mid);

        let first_tail_byte = mid_bit / 8;
        let header_len = if mid_bit % 8 != 0 {
            debug_assert_ne!(head.trailer_len, 0);
            8 - head.trailer_len
        } else {
            debug_assert_eq!(first_tail_byte, num_head_bytes);
            0
        };
        let tail = Self {
            bitmap: &self.bitmap[first_tail_byte..],
            header_len,
            trailer_len: self.trailer_len,
        };
        debug_assert_eq!(tail.len(), self.len() - mid);

        (head, tail)
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

            prop_assert_eq!(validity.get_cloned(index), bits.get(index).copied());
            let index_res = std::panic::catch_unwind(|| validity.at(index));
            prop_assert_eq!(index_res.ok(), bits.get(index).copied());
            if in_bounds {
                prop_assert_eq!(unsafe { validity.get_cloned_unchecked(index) }, bits[index]);
            }
        }

        #[test]
        fn split_at(((bitmap, array_len), bits, index) in bitmap_bits_index()) {
            let in_bounds = index <= bits.len();
            let validity = ValiditySlice::new(&bitmap, array_len);
            prop_assert_eq!(validity, &bits[..]);

            let res = std::panic::catch_unwind(|| validity.split_at(index));

            if !in_bounds {
                prop_assert!(res.is_err());
                return Ok(());
            }

            prop_assert!(res.is_ok());
            let (validity_head, validity_tail) = res.unwrap();
            let (bits_head, bits_tail) = bits.split_at(index);
            prop_assert_eq!(validity_head, bits_head);
            prop_assert_eq!(validity_tail, bits_tail);
        }
    }
}
