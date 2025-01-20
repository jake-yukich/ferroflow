use super::ComputeBackend;
use crate::error::{Result, FerroFlowError};
use std::sync::Arc;

#[derive(Debug)]
pub struct CPUContext;

impl CPUContext {
    pub fn new() -> Self {
        Self
    }
}

pub struct CPUBackend;

impl ComputeBackend for CPUBackend {
    type Buffer = Vec<f32>;
    type Context = CPUContext;

    fn new() -> Result<Arc<Self::Context>> {
        Ok(Arc::new(CPUContext::new()))
    }

    fn allocate_buffer(
        _ctx: &Self::Context,
        size: usize,
        data: Option<&[f32]>
    ) -> Result<Self::Buffer> {
        match data {
            Some(data) => Ok(data.to_vec()),
            None => Ok(vec![0.0; size])
        }
    }

    fn read_buffer(_ctx: &Self::Context, buffer: &Self::Buffer) -> Result<Vec<f32>> {
        Ok(buffer.clone())
    }

    fn element_wise_add(
        _ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        size: usize
    ) -> Result<Self::Buffer> {
        if a.len() != size || b.len() != size {
            return Err(FerroFlowError::BufferError("Buffer size mismatch".into()));
        }
        
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    fn element_wise_multiply(
        _ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        size: usize
    ) -> Result<Self::Buffer> {
        if a.len() != size || b.len() != size {
            return Err(FerroFlowError::BufferError("Buffer size mismatch".into()));
        }
        
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
    }

    fn scalar_multiply(
        _ctx: &Self::Context,
        input: &Self::Buffer,
        scalar: f32,
        size: usize
    ) -> Result<Self::Buffer> {
        if input.len() != size {
            return Err(FerroFlowError::BufferError("Buffer size mismatch".into()));
        }
        
        Ok(input.iter().map(|x| x * scalar).collect())
    }

    fn synchronize(_ctx: &Self::Context) -> Result<()> {
        Ok(())
    }

    fn matmul(
        _ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        m: usize,
        n: usize,
        k: usize
    ) -> Result<Self::Buffer> {
        let mut c = vec![0.0; m * n];
        
        // Basic matrix multiplication
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        
        Ok(c)
    }

    fn matmul_batched(
        _ctx: &Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        batch_size: usize,
        m: usize,
        n: usize,
        k: usize
    ) -> Result<Self::Buffer> {
        let mut c = vec![0.0; batch_size * m * n];
        
        for batch in 0..batch_size {
            let batch_offset_a = batch * m * k;
            let batch_offset_b = batch * k * n;
            let batch_offset_c = batch * m * n;
            
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for kk in 0..k {
                        sum += a[batch_offset_a + i * k + kk] * b[batch_offset_b + kk * n + j];
                    }
                    c[batch_offset_c + i * n + j] = sum;
                }
            }
        }
        
        Ok(c)
    }
} 