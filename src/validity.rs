//! Strongly typed array validity bitmaps

use std::iter::{FusedIterator, Take};

/// Strongly typed view of an Arrow validity bitmap
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
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

/// Iterator over the elements of an arrow validity slice
pub type Iter<'slice> = Take<BitmapIter<'slice>>;

/// Iterator over the elements of an arrow bitmap
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
        let byte = self.current_byte?;
        let result = (byte & self.bit) != 0;

        // Move to the next byte/bit
        self.bit = self.bit.wrapping_shl(1);
        if self.bit == 1 {
            self.current_byte = self.bytes.next().copied();
        }

        // Return the result
        Some(result)
    }
}
