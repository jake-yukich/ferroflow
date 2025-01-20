use thiserror::Error;

#[derive(Error, Debug)]
pub enum FerroFlowError {
    #[error("Metal device error: {0}")]
    MetalError(String),
    
    #[error("CPU backend error: {0}")]
    CPUError(String),
    
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    #[error("Initialization error: {0}")]
    InitError(String),
    
    #[error("Buffer error: {0}")]
    BufferError(String),
}

pub type Result<T> = std::result::Result<T, FerroFlowError>; 