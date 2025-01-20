use ferroflow::{
    init_logging,
    compute::{CPUBackend, MetalBackend, ComputeBackend},
    tensor::{Tensor, Shape}
};
use std::sync::Arc;
use std::time::Instant;

fn print_matrix(data: &[f32], rows: usize, cols: usize) {
    println!("[");
    for i in 0..rows {
        print!("  ");
        for j in 0..cols {
            print!("{:6.1}", data[i * cols + j]);
            if j < cols - 1 { print!(", "); }
        }
        println!();
    }
    println!("]");
}

fn test_transposed_matmul<B: ComputeBackend>(name: &str) {
    println!("\n=== Testing {} transposed matrix multiplication ===", name);
    let ctx = B::new().unwrap();
    
    // Test case 1: A transposed
    println!("\nTest case 1: Matrix A transposed");
    {
        // Create a 2x3 matrix stored as 3x2 and transpose it during multiplication
        let a_data = vec![
            1.0, 4.0,  // Will be [1.0, 2.0, 3.0]
            2.0, 5.0,  //      [4.0, 5.0, 6.0]
            3.0, 6.0,
        ];
        
        let b_data = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ];
        
        let a = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new(vec![3, 2]),  // Stored as 3x2
            &a_data,
        ).unwrap();
        
        let b = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new(vec![3, 2]),
            &b_data,
        ).unwrap();

        println!("Matrix A (stored as 3x2, will be transposed to 2x3):");
        print_matrix(&a_data, 3, 2);
        println!("Matrix B (3x2):");
        print_matrix(&b_data, 3, 2);

        let start = Instant::now();
        // Use transpose_a=true to interpret A as 2x3
        let c = a.matmul_transposed(&b, true, false).unwrap();
        let duration = start.elapsed();
        
        let result = c.data().unwrap();
        println!("\nResult (2x2) - took {:?}:", duration);
        print_matrix(&result, 2, 2);
    }

    // Test case 2: Both matrices transposed
    println!("\nTest case 2: Both matrices transposed");
    {
        let a_data = vec![
            1.0, 3.0, 5.0,  // Will be [1.0, 2.0]
            2.0, 4.0, 6.0,  //         [3.0, 4.0]
                           //         [5.0, 6.0]
        ];
        
        let b_data = vec![
            1.0, 3.0,  // Will be [1.0, 2.0]
            2.0, 4.0,  //         [3.0, 4.0]
        ];
        
        let a = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new(vec![2, 3]),
            &a_data,
        ).unwrap();
        
        let b = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new(vec![2, 2]),
            &b_data,
        ).unwrap();

        println!("Matrix A (stored as 2x3, will be transposed to 3x2):");
        print_matrix(&a_data, 2, 3);
        println!("Matrix B (stored as 2x2, will be transposed to 2x2):");
        print_matrix(&b_data, 2, 2);

        let start = Instant::now();
        let c = a.matmul_transposed(&b, true, true).unwrap();
        let duration = start.elapsed();
        
        let result = c.data().unwrap();
        println!("\nResult (3x2) - took {:?}:", duration);
        print_matrix(&result, 3, 2);
    }
}

fn main() {
    init_logging();
    
    test_transposed_matmul::<CPUBackend>("CPU");
    test_transposed_matmul::<MetalBackend>("GPU");
} 