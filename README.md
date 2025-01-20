# FerroFlow

A deep learning framework powered by Metal GPU acceleration, written in Rust and Metal, to explore ML nuts and bolts while leveraging my Apple Silicon chip. *Work in progress.*

## Features

### Matrix Operations
- ✅ Basic matrix multiplication
- ✅ Tiled GPU matrix multiplication for better performance
- ✅ Batched matrix multiplication
- ✅ Transposed matrix operations
- ✅ Basic tiled/non-tiled implementation switching based on matrix size

### Element-wise Operations
- ✅ Addition
- ✅ Multiplication
- ✅ Scalar multiplication

### Backend Support
- ✅ Metal GPU backend with optimized compute shaders
- ✅ CPU backend for comparison and fallback
- ✅ Generic backend trait for extensibility

### Performance Optimizations
- ✅ Tiled matrix multiplication for GPU
- ✅ Efficient memory management
- ✅ Batched operations support
- ✅ Zero-copy data transfers where possible
- [ ] Optimized tile size selection

## Usage Examples

### Operator Overloading
```rust
// Matrix multiplication using operator
let c = &a * &b?;

// Addition
let sum = &a + &b?;

// Element-wise multiplication
let prod = &a & &b?;

// Scalar multiplication
let scaled = &a * 2.0?;

// Negation
let negated = -&a?;
```

### Method Chaining
```rust
// Chain multiple operations
let result = a.chain()
    .matmul(&b)
    .add(&c)
    .scalar_multiply(0.5)
    .finish()?;
```

### Transposition
```rust
// Using the transpose operator
let c = &a.t() * &b?;  // Transpose A then multiply
let d = &a * &b.t()?;  // Multiply A with transposed B

// Multiple transpositions
let e = &a.t() * &b.t()?;  // Both matrices transposed
```

### Basic Matrix Multiplication
```rust
use ferroflow::{init_logging, compute::MetalBackend, tensor::{Tensor, Shape}};
use std::sync::Arc;

// Initialize the GPU context
let ctx = MetalBackend::new()?;

// Create two matrices
let a = Tensor::<MetalBackend>::new(
    Arc::clone(&ctx),
    Shape::new(vec![2, 3]),
    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;

let b = Tensor::<MetalBackend>::new(
    Arc::clone(&ctx),
    Shape::new(vec![3, 2]),
    &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
)?;

// Multiply matrices
let c = a.matmul(&b)?;
```

### Transposed Matrix Multiplication
```rust
// Multiply matrices with transposition
let c = a.matmul_transposed(&b, true, false)?; // Transpose A
```

### Batched Matrix Multiplication
```rust
// Create batched matrices
let a = Tensor::<MetalBackend>::new(
    Arc::clone(&ctx),
    Shape::new_batched(2, 2, 3), // batch_size=2, rows=2, cols=3
    &data
)?;

// Perform batched multiplication
let c = a.matmul_batched(&b)?;
```

## Performance

The framework automatically selects the best implementation based on matrix size:
- Small matrices: Basic implementation
- Large matrices: Tiled implementation for better cache utilization
- Multiple matrices: Batched implementation for parallel processing

## Building from Source

Clone the repository and build:
```bash
git clone https://github.com/yourusername/ferroflow.git
cd ferroflow
cargo build --release
```

## Requirements
- macOS with Metal support
- Rust 1.x

## Roadmap
- [ ] Convolution operations
- [ ] Activation functions
- [ ] Automatic differentiation
- [ ] More performance optimizations
