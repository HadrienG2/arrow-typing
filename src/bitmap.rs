//! Arrow-style bitmaps

use crate::element::Slice;
use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
    iter::{FusedIterator, Take},
};

/// A bit-packed slice of booleans
///
/// This type is logically equivalent to `&[bool]`, but is implemented over a
/// bit-packed `&[u8]` representation. It is notably used to provide in-place
/// access to the null buffer/validity bitmap of Arrow arrays.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bitmap<'array> {
    /// Raw bitmap, possibly containing superfluous bits
    raw: &'array [u8],

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
impl<'array> Bitmap<'array> {
    /// Decode a bitmap from arrow-rs
    ///
    /// # Panics
    ///
    /// Panics if `array_len` is not in the expected `(bitmap.len() - 1) * 8..
    /// bitmap.len() * 8 range`.
    pub(crate) fn new(raw: &'array [u8], array_len: usize) -> Self {
        let error = "bitmap and array length don't match";
        let trailer_len = (raw.len() * 8).checked_sub(array_len).expect(error);
        assert!(trailer_len < 8, "{error}");
        Self {
            raw,
            header_len: 0,
            trailer_len: trailer_len as u8,
        }
    }

    crate::inherent_slice_methods!(element: bool, iter_lifetime: 'array);
}
//
impl Eq for Bitmap<'_> {}
//
impl Hash for Bitmap<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len());
        for b in self {
            <bool as Hash>::hash(&b, state);
        }
    }
}
//
impl<'slice> IntoIterator for &'slice Bitmap<'slice> {
    type Item = bool;
    type IntoIter = Iter<'slice>;
    fn into_iter(self) -> Self::IntoIter {
        let mut bytes = self.raw.iter();
        let current_byte = bytes.next().copied();
        (Bits {
            bytes,
            current_byte,
            bit: 1 << self.header_len,
        })
        .take(self.len())
    }
}
//
impl Ord for Bitmap<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}
//
impl<OtherBools: Slice<Element = bool>> PartialEq<OtherBools> for Bitmap<'_> {
    fn eq(&self, other: &OtherBools) -> bool {
        self.iter().eq(other.iter_cloned())
    }
}
//
impl<OtherBools: Slice<Element = bool>> PartialOrd<OtherBools> for Bitmap<'_> {
    fn partial_cmp(&self, other: &OtherBools) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter_cloned())
    }
}
//
impl Slice for Bitmap<'_> {
    type Element = bool;

    #[inline]
    fn is_consistent(&self) -> bool {
        true
    }

    #[inline]
    fn len(&self) -> usize {
        self.raw.len() * 8 - (self.header_len + self.trailer_len) as usize
    }

    #[inline]
    unsafe fn get_cloned_unchecked(&self, index: usize) -> bool {
        let bit = index + self.header_len as usize;
        self.raw.get_unchecked(bit / 8) & (1 << (bit % 8)) != 0
    }

    fn iter_cloned(&self) -> impl Iterator<Item = bool> + '_ {
        let mut bytes = self.raw.iter();
        let current_byte = bytes.next().copied();
        (Bits {
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
        let head_bitmap = &self.raw[..num_head_bytes];
        let head = Self {
            raw: head_bitmap,
            header_len: self.header_len,
            trailer_len: (head_bitmap.len() * 8 - mid_bit) as u8,
        };
        debug_assert_eq!(head.len(), mid);

        let first_tail_byte = mid_bit / 8;
        let header_len = if mid_bit % 8 == 0 {
            debug_assert_eq!(first_tail_byte, num_head_bytes);
            0
        } else {
            debug_assert_ne!(head.trailer_len, 0);
            8 - head.trailer_len
        };
        let tail = Self {
            raw: &self.raw[first_tail_byte..],
            header_len,
            trailer_len: self.trailer_len,
        };
        debug_assert_eq!(tail.len(), self.len() - mid);

        (head, tail)
    }
}

/// Iterator over the elements of a [`Bitmap`]
pub type Iter<'slice> = Take<Bits<'slice>>;

/// Iterator over the bits of an `&[u8]`
///
/// For storage efficiency reasons, Arrow bit-packs arrays of booleans into
/// `&[u8]` slices. This iterator lets you iterate over the booleans packed
/// inside of such a slice.
#[derive(Clone, Debug, Default)]
pub struct Bits<'bytes> {
    /// Iterator over the bitmap's bytes
    bytes: std::slice::Iter<'bytes, u8>,

    /// Last byte obtained from the `slice` iterator
    current_byte: Option<u8>,

    /// Currently targeted bit within current_byte
    bit: u8,
}
//
impl FusedIterator for Bits<'_> {}
//
impl<'bytes> Iterator for Bits<'bytes> {
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

    /// Generate bitmap building blocks
    fn building_blocks() -> impl Strategy<Value = (Vec<u8>, usize)> {
        prop_oneof![
            any::<Vec<bool>>().prop_map(|bits| { bits_to_bitmap(&bits) }),
            any::<(Vec<u8>, usize)>()
        ]
    }

    /// Convert bits into bitmap building blocks
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
        fn init((raw_bitmap, array_len) in building_blocks()) {
            let res = std::panic::catch_unwind(|| Bitmap::new(&raw_bitmap, array_len));

            if raw_bitmap.len() != array_len.div_ceil(8) {
                prop_assert!(res.is_err());
                return Ok(());
            }
            prop_assert!(res.is_ok());
            let bitmap = res.unwrap();

            prop_assert_eq!(bitmap.len(), array_len);
            prop_assert_eq!(bitmap.is_empty(), array_len == 0);
            prop_assert_eq!(bitmap.iter().count(), array_len);
            for (idx, bit) in bitmap.iter().enumerate() {
                prop_assert_eq!(bit, raw_bitmap[idx / 8] & (1 << (idx % 8)) != 0);
            }
        }
    }

    /// Generate a bitmap, its unpacked bits, and an index into it
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
        fn index(((raw_bitmap, array_len), bits, index) in bitmap_bits_index()) {
            let in_bounds = index < bits.len();
            let bitmap = Bitmap::new(&raw_bitmap, array_len);
            prop_assert_eq!(bitmap, &bits[..]);

            prop_assert_eq!(bitmap.get_cloned(index), bits.get(index).copied());
            let index_res = std::panic::catch_unwind(|| bitmap.at(index));
            prop_assert_eq!(index_res.ok(), bits.get(index).copied());
            if in_bounds {
                prop_assert_eq!(unsafe { bitmap.get_cloned_unchecked(index) }, bits[index]);
            }
        }

        #[test]
        fn split_at(((raw_bitmap, array_len), bits, index) in bitmap_bits_index()) {
            let in_bounds = index <= bits.len();
            let bitmap = Bitmap::new(&raw_bitmap, array_len);
            prop_assert_eq!(bitmap, &bits[..]);

            let res = std::panic::catch_unwind(|| bitmap.split_at(index));

            if !in_bounds {
                prop_assert!(res.is_err());
                return Ok(());
            }

            prop_assert!(res.is_ok());
            let (bitmap_head, bitmap_tail) = res.unwrap();
            let (bits_head, bits_tail) = bits.split_at(index);
            prop_assert_eq!(bitmap_head, bits_head);
            prop_assert_eq!(bitmap_tail, bits_tail);
        }
    }
}
