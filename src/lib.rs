//! FerroFlow: A deep learning framework powered by Metal
//! 
//! This library provides a tensor computation framework that can leverage both CPU and Metal GPU
//! backends for accelerated machine learning operations.

pub mod tensor;
pub mod compute;
pub mod metal;
pub mod error;

pub use tensor::Tensor;
pub use compute::{ComputeBackend, CPUBackend, MetalBackend};

use tracing_subscriber::{fmt, EnvFilter};

pub fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
        
    fmt()
        .with_env_filter(filter)
        .with_thread_ids(true)
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .init();
} 