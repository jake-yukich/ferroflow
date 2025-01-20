use ferroflow::{
    init_logging,
    compute::{CPUBackend, MetalBackend, ComputeBackend},
    tensor::{Tensor, Shape}
};
use std::sync::Arc;
use std::time::Instant;

fn run_ops<B: ComputeBackend>(name: &str) {
    println!("\nRunning {} operations:", name);
    let ctx = B::new().unwrap();
    
    let a = Tensor::<B>::new(
        Arc::clone(&ctx),
        Shape::new(vec![2, 3]),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ).unwrap();
    
    let b = Tensor::<B>::new(
        Arc::clone(&ctx),
        Shape::new(vec![2, 3]),
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ).unwrap();

    println!("\nInput tensors:");
    println!("a = {:?}", a.data().unwrap());
    println!("b = {:?}", b.data().unwrap());

    // Element-wise addition
    let start = Instant::now();
    let c = a.add(&b).unwrap();
    println!("\nAddition (took {:?}):", start.elapsed());
    println!("a + b = {:?}", c.data().unwrap());

    // Element-wise multiplication
    let start = Instant::now();
    let d = a.multiply(&b).unwrap();
    println!("\nMultiplication (took {:?}):", start.elapsed());
    println!("a * b = {:?}", d.data().unwrap());

    // Scalar multiplication
    let scalar = 2.0;
    let start = Instant::now();
    let e = a.scalar_multiply(scalar).unwrap();
    println!("\nScalar multiplication (took {:?}):", start.elapsed());
    println!("a * {} = {:?}", scalar, e.data().unwrap());
}

fn main() {
    init_logging();
    run_ops::<CPUBackend>("CPU");
    run_ops::<MetalBackend>("GPU");
} 