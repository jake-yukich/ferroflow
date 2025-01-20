use ferroflow::{
    init_logging,
    compute::{CPUBackend, MetalBackend, ComputeBackend},
    tensor::{Tensor, Shape}
};
use std::sync::Arc;

fn test_operators<B: ComputeBackend>(name: &str) {
    println!("\n=== Testing {} operator overloading ===", name);
    let ctx = B::new().unwrap();
    
    let a = Tensor::<B>::new(
        Arc::clone(&ctx),
        Shape::new(vec![2, 2]),
        &[1.0, 2.0, 3.0, 4.0],
    ).unwrap();
    
    // Test negation
    let neg_a = (-&a).unwrap();
    let neg_data = neg_a.data().unwrap();
    println!("Original matrix:");
    println!("[{}, {}]\n[{}, {}]", 1.0, 2.0, 3.0, 4.0);
    println!("\nNegated matrix:");
    println!("[{}, {}]\n[{}, {}]", neg_data[0], neg_data[1], neg_data[2], neg_data[3]);
    
    // Test other operators...
}

fn main() {
    init_logging();
    test_operators::<CPUBackend>("CPU");
    test_operators::<MetalBackend>("GPU");
} 