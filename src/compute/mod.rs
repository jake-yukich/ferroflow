use std::sync::Arc;
use crate::error::Result;

/// Trait representing the capabilities required for a compute backend.
pub trait ComputeBackend: Send + Sync + 'static {
    /// The buffer type used by this backend
    type Buffer: Send + Sync;
    /// The context type used by this backend
    type Context: Send + Sync + std::fmt::Debug;

    /// Creates a new instance of the compute backend
    fn new() -> Result<Arc<Self::Context>>;

    /// Allocates a buffer of the given size (in elements)
    fn allocate_buffer(ctx: &Self::Context, size: usize, data: Option<&[f32]>) -> Result<Self::Buffer>;

    /// Reads data from the buffer into a Vec<f32>
    fn read_buffer(ctx: &Self::Context, buffer: &Self::Buffer) -> Result<Vec<f32>>;

    /// Performs element-wise addition
    fn element_wise_add(
        ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        size: usize
    ) -> Result<Self::Buffer>;

    /// Performs element-wise multiplication
    fn element_wise_multiply(
        ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        size: usize
    ) -> Result<Self::Buffer>;

    /// Performs scalar multiplication
    fn scalar_multiply(
        ctx: &Self::Context,
        input: &Self::Buffer,
        scalar: f32,
        size: usize
    ) -> Result<Self::Buffer>;

    /// Synchronizes the backend (if needed)
    fn synchronize(ctx: &Self::Context) -> Result<()>;

    /// Performs matrix multiplication C = A * B with optional transposition
    /// 
    /// # Arguments
    /// * `ctx` - The compute context
    /// * `a` - Left matrix (M x K) or (K x M) if transposed
    /// * `b` - Right matrix (K x N) or (N x K) if transposed
    /// * `m` - Number of rows in result
    /// * `n` - Number of columns in result
    /// * `k` - Inner dimension
    /// * `transpose_a` - Whether to transpose matrix A
    /// * `transpose_b` - Whether to transpose matrix B
    fn matmul_transposed(
        ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
        transpose_a: bool,
        transpose_b: bool
    ) -> Result<Self::Buffer>;

    /// Performs batched matrix multiplication with optional transposition
    fn matmul_transposed_batched(
        ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        batch_size: usize,
        m: usize,
        n: usize,
        k: usize,
        transpose_a: bool,
        transpose_b: bool
    ) -> Result<Self::Buffer>;

    // Default implementations that call non-transposed versions
    fn matmul(ctx: &Self::Context, a: &Self::Buffer, b: &Self::Buffer,
        m: usize, n: usize, k: usize) -> Result<Self::Buffer> {
        Self::matmul_transposed(ctx, a, b, m, n, k, false, false)
    }

    fn matmul_batched(ctx: &Self::Context, a: &Self::Buffer, b: &Self::Buffer,
        batch_size: usize, m: usize, n: usize, k: usize) -> Result<Self::Buffer> {
        Self::matmul_transposed_batched(ctx, a, b, batch_size, m, n, k, false, false)
    }
}

mod cpu;
mod metal;

pub use cpu::CPUBackend;
pub use metal::MetalBackend; 