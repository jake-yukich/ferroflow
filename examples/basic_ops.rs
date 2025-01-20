use ferroflow::{init_logging, compute::{MetalBackend, ComputeBackend}, tensor::{Tensor, Shape}};
use std::sync::Arc;

fn main() {
    init_logging();

    let ctx = MetalBackend::new().unwrap();
    
    let a = Tensor::<MetalBackend>::new(
        Arc::clone(&ctx),
        Shape::new(vec![2, 2]),
        &[1.0, 2.0, 3.0, 4.0],
    ).unwrap();
    
    let b = Tensor::<MetalBackend>::new(
        Arc::clone(&ctx),
        Shape::new(vec![2, 2]),
        &[5.0, 6.0, 7.0, 8.0],
    ).unwrap();

    println!("a = {:?}", a.data().unwrap());
    println!("b = {:?}", b.data().unwrap());
    
    let c = a.add(&b).unwrap();
    println!("a + b = {:?}", c.data().unwrap());
} 