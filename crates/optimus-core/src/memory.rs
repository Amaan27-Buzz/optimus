use crate::{Error, Result};

/// Logical reservations only. Does NOT enforce a physical or OS memory limit.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    limit: usize,
    used: usize,
    peak: usize,
}

impl MemoryBudget {
    pub fn new(limit: usize) -> Self { Self { limit, used: 0, peak: 0 } }
    pub fn used_bytes(&self) -> usize { self.used }
    pub fn peak_bytes(&self) -> usize { self.peak }
    pub fn available_bytes(&self) -> usize { self.limit - self.used }

    pub fn reserve(&mut self, bytes: usize) -> Result<()> {
        if bytes > self.available_bytes() {
            return Err(Error::BudgetExceeded { requested: bytes, available: self.available_bytes() });
        }
        self.used += bytes;
        self.peak = self.peak.max(self.used);
        Ok(())
    }
    pub fn release(&mut self, bytes: usize) -> Result<()> {
        if bytes > self.used { return Err(Error::InvalidArgument("release exceeds reservation")); }
        self.used -= bytes;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accounting() {
        let mut b = MemoryBudget::new(128);
        b.reserve(96).unwrap();
        assert!(matches!(b.reserve(33), Err(Error::BudgetExceeded { .. })));
        b.release(80).unwrap();
        b.reserve(48).unwrap();
        assert_eq!(b.used_bytes(), 64);
        assert_eq!(b.peak_bytes(), 96);
        assert!(b.release(100).is_err());
    }
}
