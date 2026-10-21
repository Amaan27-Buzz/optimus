use crate::{Error, Result};

/// Signed two's-complement INT4, low nibble first, one scale per group.
/// A reference primitive, NOT a GGUF/OpenVINO/Safetensors format decoder.
pub fn dequantize_int4(packed: &[u8], scales: &[f32], group_size: usize, count: usize) -> Result<Vec<f32>> {
    if group_size == 0 { return Err(Error::InvalidArgument("group size cannot be zero")); }
    let bytes = count.checked_add(1).ok_or(Error::ArithmeticOverflow)? / 2;
    let groups = count.checked_add(group_size - 1).ok_or(Error::ArithmeticOverflow)? / group_size;
    if packed.len() != bytes || scales.len() != groups || scales.iter().any(|s| !s.is_finite()) {
        return Err(Error::InvalidArgument("invalid packed data or scales"));
    }
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let byte = packed[i / 2];
        let nibble = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
        let value = if nibble >= 8 { nibble as i8 - 16 } else { nibble as i8 };
        result.push(value as f32 * scales[i / group_size]);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_nibbles() {
        assert_eq!(dequantize_int4(&[0x7f], &[0.5], 2, 2).unwrap(), vec![-0.5, 3.5]);
    }
    #[test]
    fn different_scales_odd_count() {
        assert_eq!(dequantize_int4(&[0x21, 0x03], &[1., 2.], 2, 3).unwrap(), vec![1., 2., 6.]);
    }
    #[test]
    fn invalid_input() {
        assert!(dequantize_int4(&[], &[1.], 2, 2).is_err());
        assert!(dequantize_int4(&[1], &[1.], 0, 2).is_err());
    }
}
