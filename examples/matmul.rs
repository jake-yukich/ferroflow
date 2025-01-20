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

// Helper function to compute expected result
fn compute_expected_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut result = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    result
}

// Helper function to verify results
fn verify_result(actual: &[f32], expected: &[f32], epsilon: f32) -> bool {
    if actual.len() != expected.len() {
        println!("❌ Length mismatch: actual {} vs expected {}", actual.len(), expected.len());
        return false;
    }
    
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if (a - e).abs() > epsilon {
            println!("❌ Mismatch at index {}: actual {} vs expected {}", i, a, e);
            return false;
        }
    }
    true
}

fn test_matmul<B: ComputeBackend>(name: &str) {
    println!("\n=== Testing {} matrix multiplication ===", name);
    let ctx = B::new().unwrap();
    let mut all_tests_passed = true;
    
    // Test 1: Small matrix (2x3 × 3x2)
    {
        println!("\nTest 1: Small matrix multiplication (2x3 × 3x2)");
        let a_data = vec![1.0, 2.0, 3.0,
                         4.0, 5.0, 6.0];
        let b_data = vec![7.0, 8.0,
                         9.0, 10.0,
                         11.0, 12.0];
        
        let expected = compute_expected_matmul(&a_data, &b_data, 2, 3, 2);
        
        let a = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new(vec![2, 3]),
            &a_data,
        ).unwrap();
        
        let b = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new(vec![3, 2]),
            &b_data,
        ).unwrap();

        println!("Matrix A (2x3):");
        print_matrix(&a_data, 2, 3);
        println!("Matrix B (3x2):");
        print_matrix(&b_data, 3, 2);

        let start = Instant::now();
        let c = a.matmul(&b).unwrap();
        let duration = start.elapsed();
        
        let result = c.data().unwrap();
        println!("Result (2x2) - took {:?}:", duration);
        print_matrix(&result, 2, 2);
        
        let test_passed = verify_result(&result, &expected, 1e-5);
        println!("{}", if test_passed { "✅ Test passed!" } else { "❌ Test failed!" });
        all_tests_passed &= test_passed;
    }

    // Test 2: Medium matrix (4x3 × 3x4)
    {
        println!("\nTest 2: Medium matrix multiplication (4x3 × 3x4)");
        let a_data = vec![1.0, 2.0, 3.0,
                         4.0, 5.0, 6.0,
                         7.0, 8.0, 9.0,
                         10.0, 11.0, 12.0];
        let b_data = vec![1.0, 2.0, 3.0, 4.0,
                         5.0, 6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0, 12.0];
        
        let expected = compute_expected_matmul(&a_data, &b_data, 4, 3, 4);
        
        let a = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new(vec![4, 3]),
            &a_data,
        ).unwrap();
        
        let b = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new(vec![3, 4]),
            &b_data,
        ).unwrap();

        println!("Matrix A (4x3):");
        print_matrix(&a_data, 4, 3);
        println!("Matrix B (3x4):");
        print_matrix(&b_data, 3, 4);

        let start = Instant::now();
        let c = a.matmul(&b).unwrap();
        let duration = start.elapsed();
        
        let result = c.data().unwrap();
        println!("Result (4x4) - took {:?}:", duration);
        print_matrix(&result, 4, 4);
        
        let test_passed = verify_result(&result, &expected, 1e-5);
        println!("{}", if test_passed { "✅ Test passed!" } else { "❌ Test failed!" });
        all_tests_passed &= test_passed;
    }

    // Test 3: Larger matrix (32x32 × 32x32)
    {
        println!("\nTest 3: Larger matrix multiplication (32x32 × 32x32)");
        let data_a: Vec<f32> = (0..32*32).map(|x| x as f32).collect();
        let data_b: Vec<f32> = (0..32*32).map(|x| (x + 1) as f32).collect();
        
        let expected = compute_expected_matmul(&data_a, &data_b, 32, 32, 32);

        let a = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new(vec![32, 32]),
            &data_a,
        ).unwrap();
        
        let b = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new(vec![32, 32]),
            &data_b,
        ).unwrap();

        println!("Matrix A and B: 32x32 matrices (data too large to display)");

        let start = Instant::now();
        let c = a.matmul(&b).unwrap();
        let duration = start.elapsed();
        
        let result = c.data().unwrap();
        println!("Result (32x32) - took {:?}", duration);
        println!("First few elements of result: {:?}", &result[0..4]);
        
        let test_passed = verify_result(&result, &expected, 1e-5);
        println!("{}", if test_passed { "✅ Test passed!" } else { "❌ Test failed!" });
        all_tests_passed &= test_passed;
    }

    println!("\n=== Overall Result ===");
    println!("{}", if all_tests_passed {
        "✅ All tests passed!"
    } else {
        "❌ Some tests failed!"
    });
}

fn main() {
    init_logging();
    
    test_matmul::<CPUBackend>("CPU");
    test_matmul::<MetalBackend>("GPU");
} 