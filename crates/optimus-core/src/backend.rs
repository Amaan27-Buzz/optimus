use crate::{Result, Tensor};

/// Separates backend choice from tensor math; only CPU currently works.
pub trait MatmulBackend {
    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>;
}

#[derive(Debug, Default)]
pub struct CpuReference;

impl MatmulBackend for CpuReference {
    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        lhs.matmul(rhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind { Cpu, IntelNpu, NvidiaGpu }

impl DeviceKind {
    pub fn implemented(self) -> bool { matches!(self, Self::Cpu) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn device_status_is_honest() {
        assert!(DeviceKind::Cpu.implemented());
        assert!(!DeviceKind::IntelNpu.implemented());
        assert!(!DeviceKind::NvidiaGpu.implemented());
    }
}
