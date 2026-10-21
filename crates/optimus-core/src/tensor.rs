use crate::{Error, Result};

/// Contiguous row-major f32 tensor for numerical reference tests.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn from_vec(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        if shape.is_empty() || shape.contains(&0) {
            return Err(Error::InvalidShape("dimensions must be positive and nonempty"));
        }
        let expected = shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or(Error::ArithmeticOverflow)?;
        if expected != data.len() {
            return Err(Error::InvalidShape("shape does not match element count"));
        }
        Ok(Self { shape, data })
    }

    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn as_slice(&self) -> &[f32] { &self.data }

    /// Reference 2-D multiplication, not an optimized BLAS kernel.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err(Error::InvalidShape("matmul requires 2-D matrices"));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (other_k, n) = (rhs.shape[0], rhs.shape[1]);
        if k != other_k { return Err(Error::InvalidShape("inner dimensions do not match")); }
        let len = m.checked_mul(n).ok_or(Error::ArithmeticOverflow)?;
        let mut result = vec![0.0f32; len];
        for i in 0..m {
            for p in 0..k {
                let lhs = self.data[i * k + p];
                for j in 0..n {
                    result[i * n + j] += lhs * rhs.data[p * n + j];
                }
            }
        }
        Ok(Self { shape: vec![m, n], data: result })
    }

    /// RMSNorm along the final dimension.
    pub fn rms_norm(&self, weight: &[f32], epsilon: f32) -> Result<Self> {
        let width = *self.shape.last().ok_or(Error::InvalidShape("no dimension"))?;
        if weight.len() != width || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::InvalidArgument("invalid RMSNorm weight or epsilon"));
        }
        let mut data = self.data.clone();
        for row in data.chunks_exact_mut(width) {
            let mean_square = row.iter().map(|x| x * x).sum::<f32>() / width as f32;
            let scale = (mean_square + epsilon).sqrt().recip();
            for (value, w) in row.iter_mut().zip(weight) { *value *= scale * w; }
        }
        Ok(Self { shape: self.shape.clone(), data })
    }

    /// Stable softmax along the final dimension, requiring finite inputs.
    pub fn softmax_last_dim(&self) -> Result<Self> {
        let width = *self.shape.last().ok_or(Error::InvalidShape("no dimension"))?;
        let mut data = self.data.clone();
        for row in data.chunks_exact_mut(width) {
            if row.iter().any(|x| !x.is_finite()) {
                return Err(Error::InvalidArgument("softmax expects finite values"));
            }
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for x in row.iter_mut() { *x = (*x - max).exp(); sum += *x; }
            for x in row.iter_mut() { *x /= sum; }
        }
        Ok(Self { shape: self.shape.clone(), data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_values() {
        let a = Tensor::from_vec(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let b = Tensor::from_vec(vec![3, 2], vec![7., 8., 9., 10., 11., 12.]).unwrap();
        assert_eq!(a.matmul(&b).unwrap().as_slice(), &[58., 64., 139., 154.]);
    }
    #[test]
    fn shape_checks() {
        assert!(Tensor::from_vec(vec![2, 2], vec![1.]).is_err());
        let a = Tensor::from_vec(vec![2, 2], vec![0.; 4]).unwrap();
        let b = Tensor::from_vec(vec![3, 1], vec![0.; 3]).unwrap();
        assert!(a.matmul(&b).is_err());
    }
    #[test]
    fn stable_softmax() {
        let x = Tensor::from_vec(vec![1, 3], vec![1000., 1000., 1000.]).unwrap();
        for value in x.softmax_last_dim().unwrap().as_slice() {
            assert!((value - 1. / 3.).abs() < 1e-6);
        }
    }
    #[test]
    fn rms_norm_reference() {
        let x = Tensor::from_vec(vec![1, 2], vec![3., 4.]).unwrap();
        let normalized = x.rms_norm(&[1., 1.], 1e-6).unwrap();
        let rms = (12.5f32 + 1e-6).sqrt();
        assert!((normalized.as_slice()[0] - 3. / rms).abs() < 1e-5);
        assert!((normalized.as_slice()[1] - 4. / rms).abs() < 1e-5);
    }
}
