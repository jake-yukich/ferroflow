use std::sync::Arc;
use crate::compute::ComputeBackend;
use crate::error::{Result, FerroFlowError};
use tracing::{debug, error, instrument};
use std::ops::{Add, Mul, Neg, BitAnd};

/// Represents the shape of a tensor.
/// Implements Clone to allow easy shape reuse and Debug for better error messages.
#[derive(Debug, Clone, PartialEq)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }
    
    pub fn size(&self) -> usize {
        self.0.iter().product()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Returns the batch size if this is a batched matrix (first dimension)
    pub fn batch_size(&self) -> Option<usize> {
        if self.0.len() >= 3 {
            Some(self.0[0])
        } else {
            None
        }
    }

    /// Returns the matrix dimensions (height, width) ignoring batch
    pub fn matrix_dims(&self) -> (usize, usize) {
        match self.0.len() {
            2 => (self.0[0], self.0[1]),
            3 => (self.0[1], self.0[2]),
            _ => panic!("Invalid shape for matrix operation")
        }
    }

    /// Creates a new shape for batched matrices
    pub fn new_batched(batch: usize, rows: usize, cols: usize) -> Self {
        Self(vec![batch, rows, cols])
    }

    pub fn into_dims4(&self) -> crate::error::Result<(usize, usize, usize, usize)> {
        if self.0.len() != 4 {
            return Err(crate::error::FerroFlowError::ShapeMismatch(
                format!("Expected 4 dimensions, got {}", self.0.len())
            ));
        }
        Ok((self.0[0], self.0[1], self.0[2], self.0[3]))
    }
}

/// A generic tensor implementation that works with any compute backend.
/// B: ComputeBackend ensures the backend implements all required operations.
/// Uses Arc for thread-safe sharing of the context.
pub struct Tensor<B: ComputeBackend> {
    ctx: Arc<B::Context>,
    shape: Shape,
    pub(crate) buffer: B::Buffer,  // Make buffer accessible within crate
}

impl<B: ComputeBackend> Tensor<B> {
    /// Creates a new tensor with the given shape and data.
    /// Uses the backend's buffer allocation mechanism.
    #[instrument(skip(data))]
    pub fn new(ctx: Arc<B::Context>, shape: Shape, data: &[f32]) -> Result<Self> {
        debug!("Creating new tensor with shape {:?}", shape);
        
        if data.len() != shape.size() {
            error!("Data length {} doesn't match shape size {}", data.len(), shape.size());
            return Err(FerroFlowError::ShapeMismatch(
                format!("Data length {} doesn't match shape size {}", data.len(), shape.size())
            ));
        }
        
        let buffer = B::allocate_buffer(&ctx, shape.size(), Some(data))?;
        debug!("Successfully allocated buffer for tensor");
        
        Ok(Self {
            buffer,
            shape,
            ctx,
        })
    }
    
    /// Creates a new tensor filled with zeros.
    /// Useful for pre-allocating output tensors.
    pub fn zeros(ctx: Arc<B::Context>, shape: Shape) -> Result<Self> {
        let buffer = B::allocate_buffer(&ctx, shape.size(), None)?;
        
        Ok(Self {
            buffer,
            shape,
            ctx,
        })
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Reads the tensor data into a Vec<f32>.
    /// Useful for debugging and verification.
    pub fn data(&self) -> Result<Vec<f32>> {
        B::read_buffer(&self.ctx, &self.buffer)
    }

    /// Element-wise addition of two tensors.
    #[instrument(skip(self, other))]
    pub fn add(&self, other: &Self) -> Result<Self> {
        debug!("Adding tensors with shapes {:?} and {:?}", self.shape, other.shape);
        
        if self.shape != other.shape {
            error!("Shape mismatch in add operation");
            return Err(FerroFlowError::ShapeMismatch(
                format!("Cannot add tensors with shapes {:?} and {:?}", self.shape, other.shape)
            ));
        }

        let result_buffer = B::element_wise_add(
            &self.ctx,
            &self.buffer,
            &other.buffer,
            self.shape.size(),
        )?;
        
        debug!("Successfully completed add operation");
        
        Ok(Self {
            buffer: result_buffer,
            shape: self.shape.clone(),
            ctx: Arc::clone(&self.ctx),
        })
    }

    /// Element-wise multiplication of two tensors.
    pub fn multiply(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(FerroFlowError::ShapeMismatch(
                format!("Cannot multiply tensors with shapes {:?} and {:?}", self.shape, other.shape)
            ));
        }

        let result_buffer = B::element_wise_multiply(
            &self.ctx,
            &self.buffer,
            &other.buffer,
            self.shape.size(),
        )?;

        Ok(Self {
            buffer: result_buffer,
            shape: self.shape.clone(),
            ctx: Arc::clone(&self.ctx),
        })
    }

    /// Multiplication by a scalar value.
    pub fn scalar_multiply(&self, scalar: f32) -> Result<Self> {
        let result_buffer = B::scalar_multiply(
            &self.ctx,
            &self.buffer,
            scalar,
            self.shape.size(),
        )?;

        Ok(Self {
            buffer: result_buffer,
            shape: self.shape.clone(),
            ctx: Arc::clone(&self.ctx),
        })
    }

    /// Performs matrix multiplication with another tensor
    #[instrument(skip(self, other))]
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        // Get dimensions
        if self.shape.dims().len() < 2 || other.shape.dims().len() < 2 {
            return Err(FerroFlowError::ShapeMismatch(
                "Matmul requires at least 2D tensors".into()
            ));
        }

        let (m, k1) = self.shape.matrix_dims();
        let (k2, n) = other.shape.matrix_dims();
        
        if k1 != k2 {
            return Err(FerroFlowError::ShapeMismatch(
                format!("Incompatible dimensions for matmul: {:?} and {:?}", 
                    self.shape.dims(), other.shape.dims())
            ));
        }

        // Check if this is a batched operation
        match (self.shape.batch_size(), other.shape.batch_size()) {
            (Some(b1), Some(b2)) if b1 == b2 => {
                debug!("Performing batched matmul with shapes {:?} x {:?}", self.shape, other.shape);
                let result_buffer = B::matmul_batched(
                    &self.ctx,
                    &self.buffer,
                    &other.buffer,
                    b1,
                    m,
                    n,
                    k1
                )?;
                Ok(Self {
                    buffer: result_buffer,
                    shape: Shape::new_batched(b1, m, n),
                    ctx: Arc::clone(&self.ctx),
                })
            },
            (None, None) => {
                debug!("Performing matmul with shapes {:?} x {:?}", self.shape, other.shape);
                let result_buffer = B::matmul(
                    &self.ctx,
                    &self.buffer,
                    &other.buffer,
                    m,
                    n,
                    k1
                )?;
                Ok(Self {
                    buffer: result_buffer,
                    shape: Shape::new(vec![m, n]),
                    ctx: Arc::clone(&self.ctx),
                })
            },
            _ => Err(FerroFlowError::ShapeMismatch(
                "Batch sizes must match for batched matmul".into()
            ))
        }
    }
    
    /// Performs matrix multiplication with optional transposition
    pub fn matmul_transposed(&self, other: &Self, transpose_a: bool, transpose_b: bool) -> Result<Self> {
        let (m_a, k_a) = if transpose_a {
            (self.shape.dims()[1], self.shape.dims()[0])
        } else {
            (self.shape.dims()[0], self.shape.dims()[1])
        };
        
        let (k_b, n_b) = if transpose_b {
            (other.shape.dims()[1], other.shape.dims()[0])
        } else {
            (other.shape.dims()[0], other.shape.dims()[1])
        };
        
        if k_a != k_b {
            return Err(FerroFlowError::ShapeMismatch(
                format!("Incompatible dimensions for matrix multiplication: {}x{} and {}x{}", 
                    m_a, k_a, k_b, n_b)
            ));
        }
        
        let result_buffer = B::matmul_transposed(
            &self.ctx,
            &self.buffer,
            &other.buffer,
            m_a,
            n_b,
            k_a,
            transpose_a,
            transpose_b
        )?;
        
        Ok(Self {
            buffer: result_buffer,
            shape: Shape::new(vec![m_a, n_b]),
            ctx: Arc::clone(&self.ctx),
        })
    }

    pub fn t(&self) -> TransposedTensor<B> {
        TransposedTensor { tensor: self, transpose: true }
    }

    pub fn chain(self) -> TensorChain<B> {
        TensorChain { tensor: Ok(self) }
    }

    // Create identity matrix
    pub fn eye(ctx: Arc<B::Context>, size: usize) -> Result<Self> {
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        Self::new(ctx, Shape::new(vec![size, size]), &data)
    }

    // Create matrix filled with a specific value
    pub fn full(ctx: Arc<B::Context>, shape: Shape, value: f32) -> Result<Self> {
        let size = shape.size();
        let data = vec![value; size];
        Self::new(ctx, shape, &data)
    }

    // Create random matrix
    pub fn rand(ctx: Arc<B::Context>, shape: Shape) -> Result<Self> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size = shape.size();
        let data: Vec<f32> = (0..size).map(|_| rng.gen()).collect();
        Self::new(ctx, shape, &data)
    }

    // Add a public buffer accessor method
    pub fn buffer(&self) -> &B::Buffer {
        &self.buffer
    }
}

pub struct TransposedTensor<'a, B: ComputeBackend> {
    tensor: &'a Tensor<B>,
    transpose: bool,
}

impl<'a, B: ComputeBackend> std::ops::Mul for &'a TransposedTensor<'a, B> {
    type Output = crate::error::Result<Tensor<B>>;

    fn mul(self, other: &'a TransposedTensor<B>) -> Self::Output {
        let (m, k) = if self.transpose {
            (self.tensor.shape().dims()[1], self.tensor.shape().dims()[0])
        } else {
            (self.tensor.shape().dims()[0], self.tensor.shape().dims()[1])
        };
        
        let (k2, n) = if other.transpose {
            (other.tensor.shape().dims()[1], other.tensor.shape().dims()[0])
        } else {
            (other.tensor.shape().dims()[0], other.tensor.shape().dims()[1])
        };
        
        if k != k2 {
            return Err(crate::error::FerroFlowError::ShapeMismatch(
                format!("Incompatible dimensions for matrix multiplication: {}x{} and {}x{}", m, k, k2, n)
            ));
        }

        B::matmul_transposed(
            &self.tensor.ctx,
            &self.tensor.buffer,
            &other.tensor.buffer,
            m,
            n,
            k,
            self.transpose,
            other.transpose
        ).map(|buffer| Tensor {
            ctx: Arc::clone(&self.tensor.ctx),
            shape: Shape::new(vec![m, n]),
            buffer,
        })
    }
}

// Implement Drop to ensure proper cleanup of backend resources
impl<B: ComputeBackend> Drop for Tensor<B> {
    fn drop(&mut self) {
        // Backend-specific cleanup is handled by the backend's Buffer type
    }
}

impl<B: ComputeBackend> Add for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn add(self, other: &Tensor<B>) -> Self::Output {
        self.add(other)
    }
}

impl<B: ComputeBackend> Mul for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn mul(self, other: &Tensor<B>) -> Self::Output {
        self.matmul(other)
    }
}

pub struct TensorChain<B: ComputeBackend> {
    tensor: Result<Tensor<B>>
}

impl<B: ComputeBackend> TensorChain<B> {
    pub fn matmul(self, other: &Tensor<B>) -> Self {
        TensorChain {
            tensor: self.tensor.and_then(|t| t.matmul(other))
        }
    }
    
    // TODO: Implement transpose when backend support is complete
    // pub fn transpose(self) -> Self {
    //     TensorChain {
    //         tensor: self.tensor.and_then(|t| t.transpose())
    //     }
    // }
    
    pub fn add(self, other: &Tensor<B>) -> Self {
        TensorChain {
            tensor: self.tensor.and_then(|t| t.add(other))
        }
    }
    
    pub fn multiply(self, other: &Tensor<B>) -> Self {
        TensorChain {
            tensor: self.tensor.and_then(|t| t.multiply(other))
        }
    }
    
    pub fn scalar_multiply(self, scalar: f32) -> Self {
        TensorChain {
            tensor: self.tensor.and_then(|t| t.scalar_multiply(scalar))
        }
    }
    
    pub fn finish(self) -> Result<Tensor<B>> {
        self.tensor
    }
}

// Add scalar multiplication
impl<B: ComputeBackend> Mul<f32> for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn mul(self, scalar: f32) -> Self::Output {
        self.scalar_multiply(scalar)
    }
}

// Add element-wise multiplication
impl<B: ComputeBackend> BitAnd for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn bitand(self, other: &Tensor<B>) -> Self::Output {
        self.multiply(other)
    }
}

// Add negation
impl<B: ComputeBackend> Neg for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn neg(self) -> Self::Output {
        self.scalar_multiply(-1.0)
    }
} 