#![forbid(unsafe_code)]

//! Correctness-first tensor operations; not an optimized inference engine.

pub mod backend;
pub mod memory;
pub mod quant;
pub mod tensor;
pub use tensor::Tensor;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidShape(&'static str),
    InvalidArgument(&'static str),
    ArithmeticOverflow,
    BudgetExceeded { requested: usize, available: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape(s) => write!(f, "invalid shape: {s}"),
            Self::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            Self::ArithmeticOverflow => write!(f, "arithmetic overflow"),
            Self::BudgetExceeded { requested, available } => {
                write!(f, "memory budget exceeded: requested {requested}, available {available} bytes")
            }
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
