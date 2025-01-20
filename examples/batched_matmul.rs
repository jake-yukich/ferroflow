use ferroflow::{
    init_logging,
    compute::{CPUBackend, MetalBackend, ComputeBackend},
    tensor::{Tensor, Shape}
};
use std::sync::Arc;
use std::time::Instant;
use std::fmt::Write;

const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const BLUE: &str = "\x1b[34m";
const YELLOW: &str = "\x1b[33m";
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";

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

fn visualize_shape_mismatch(a_shape: &[usize], b_shape: &[usize]) {
    println!("\nShape mismatch visualization:");
    println!("{}A shape: [{}]{}",
        BLUE, 
        a_shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" × "),
        RESET
    );
    println!("{}B shape: [{}]{}",
        YELLOW, 
        b_shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" × "),
        RESET
    );
    
    println!("\nMatrix dimensions:");
    if a_shape.len() >= 3 && b_shape.len() >= 3 {
        println!("{}A: batch_{} × {} × {}{}", 
            BLUE, a_shape[0], a_shape[1], a_shape[2], RESET);
        println!("{}B: batch_{} × {} × {}{}", 
            YELLOW, b_shape[0], b_shape[1], b_shape[2], RESET);
        println!("   {}", if a_shape[0] != b_shape[0] { 
            format!("{}❌{}", RED, RESET) 
        } else { 
            format!("{}✓{}", GREEN, RESET) 
        });
        println!("         {}", if a_shape[2] != b_shape[1] { 
            format!("{}❌{}", RED, RESET) 
        } else { 
            format!("{}✓{}", GREEN, RESET) 
        });
        
        println!("\n{}ASCII Visualization:{}", BOLD, RESET);
        
        // Show batch dimension
        let mut batch_viz = String::new();
        writeln!(&mut batch_viz, "{}Batch dimension:{}", BOLD, RESET).unwrap();
        writeln!(&mut batch_viz, "A: [{}{}{}] batches    B: [{}{}{}] batches", 
            BLUE, "■".repeat(a_shape[0]), RESET,
            YELLOW, "■".repeat(b_shape[0]), RESET).unwrap();
        if a_shape[0] != b_shape[0] {
            writeln!(&mut batch_viz, "   {}≠{}", RED, RESET).unwrap();
        } else {
            writeln!(&mut batch_viz, "   {}={}", GREEN, RESET).unwrap();
        }
        println!("{}", batch_viz);

        // Show matrix shapes for each batch
        println!("{}Matrix shapes in each batch:{}", BOLD, RESET);
        println!("{}A matrices:{}          {}B matrices:{}", BLUE, RESET, YELLOW, RESET);
        
        let max_height = a_shape[1].max(b_shape[1]);
        let connector = if a_shape[2] == b_shape[1] { 
            format!("{}   ×   {}", GREEN, RESET) 
        } else { 
            format!("{}   ≠   {}", RED, RESET) 
        };

        // Top of boxes
        println!("{}┌{}┐{}{}┌{}┐{}{}", 
            BLUE, "─".repeat(a_shape[2] + 2), RESET,
            connector,
            "─".repeat(b_shape[2] + 2), YELLOW, RESET
        );
        
        // Middle of boxes
        for _ in 0..max_height {
            let a_spaces = " ".repeat(a_shape[2] + 2);
            let b_spaces = " ".repeat(b_shape[2] + 2);
            
            println!("{}│{}│{}{}{}│{}│{}", 
                BLUE, a_spaces, RESET,
                connector,
                YELLOW, b_spaces, RESET
            );
        }
        
        // Bottom of boxes
        println!("{}└{}┘{}{}└{}┘{}{}", 
            BLUE, "─".repeat(a_shape[2] + 2), RESET,
            connector,
            "─".repeat(b_shape[2] + 2), YELLOW, RESET
        );
        
        // Show dimension labels
        println!("\n{}Dimension breakdown:{}", BOLD, RESET);
        println!("{}A: {} batches × {} rows × {} columns{}", 
            BLUE, a_shape[0], a_shape[1], a_shape[2], RESET);
        println!("{}B: {} batches × {} rows × {} columns{}", 
            YELLOW, b_shape[0], b_shape[1], b_shape[2], RESET);
        
    } else if a_shape.len() == 2 && b_shape.len() >= 3 {
        println!("{}A: {} × {}{}", BLUE, a_shape[0], a_shape[1], RESET);
        println!("{}B: batch_{} × {} × {}{}", 
            YELLOW, b_shape[0], b_shape[1], b_shape[2], RESET);
        println!("   {}❌ (non-batched vs batched){}", RED, RESET);
        
        println!("\n{}ASCII Visualization:{}", BOLD, RESET);
        println!("{}A (single matrix):{}", BLUE, RESET);
        
        // Single matrix box
        println!("┌{}┐", "─".repeat(a_shape[1] + 2));
        for _ in 0..a_shape[0] {
            println!("│{}│", " ".repeat(a_shape[1] + 2));
        }
        println!("└{}┘", "─".repeat(a_shape[1] + 2));
        
        println!("\n{}B (batched matrices):{}", YELLOW, RESET);
        for batch in 0..b_shape[0] {
            println!("Batch {}:", batch + 1);
            println!("┌{}┐", "─".repeat(b_shape[2] + 2));
            for _ in 0..b_shape[1] {
                println!("│{}│", " ".repeat(b_shape[2] + 2));
            }
            println!("└{}┘", "─".repeat(b_shape[2] + 2));
            if batch < b_shape[0] - 1 {
                println!();
            }
        }
        
        println!("\n{}Cannot multiply: Single matrix with batched matrices{}", 
            RED, RESET);
    }
}

fn test_batched_matmul<B: ComputeBackend>(name: &str) {
    println!("\n=== Testing {} batched matrix multiplication ===", name);
    let ctx = B::new().unwrap();
    
    println!("\nTest case 1: Successful batch multiplication");
    {
        let batch_size = 2;
        let m = 2;
        let k = 3;
        let n = 2;
        
        let a_data: Vec<f32> = vec![
            // Batch 1
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            // Batch 2
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ];
        
        let b_data: Vec<f32> = vec![
            // Batch 1
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            // Batch 2
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ];
        
        let a = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new_batched(batch_size, m, k),
            &a_data,
        ).unwrap();
        
        let b = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new_batched(batch_size, k, n),
            &b_data,
        ).unwrap();

        println!("Input tensors:");
        println!("Batch 1, Matrix A (2x3):");
        print_matrix(&a_data[0..6], m, k);
        println!("Batch 1, Matrix B (3x2):");
        print_matrix(&b_data[0..6], k, n);
        println!("Batch 2, Matrix A (2x3):");
        print_matrix(&a_data[6..12], m, k);
        println!("Batch 2, Matrix B (3x2):");
        print_matrix(&b_data[6..12], k, n);

        let start = Instant::now();
        let c = a.matmul(&b).unwrap();
        let duration = start.elapsed();
        
        let result = c.data().unwrap();
        println!("\nResults - took {:?}:", duration);
        println!("Batch 1 Result (2x2):");
        print_matrix(&result[0..4], m, n);
        println!("Batch 2 Result (2x2):");
        print_matrix(&result[4..8], m, n);
    }

    println!("\nTest case 2: Mismatched batch sizes");
    {
        let a_shape = Shape::new_batched(2, 2, 3);
        let b_shape = Shape::new_batched(3, 3, 2);
        
        let a = Tensor::<B>::new(
            Arc::clone(&ctx),
            a_shape.clone(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
              7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ).unwrap();
        
        let b = Tensor::<B>::new(
            Arc::clone(&ctx),
            b_shape.clone(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
              7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
              13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        ).unwrap();

        println!("Attempting matmul with mismatched batch sizes:");
        visualize_shape_mismatch(a_shape.dims(), b_shape.dims());
        
        match a.matmul(&b) {
            Ok(_) => println!("❌ Error: Expected failure for mismatched batch sizes but got success"),
            Err(e) => {
                println!("✅ Successfully caught error: {}", e);
                println!("\nProblem: Batch sizes must match (2 ≠ 3)");
                println!("Solution: Ensure both tensors have the same batch size");
            }
        }
    }

    println!("\nTest case 3: Non-batched with batched");
    {
        let a_shape = Shape::new(vec![2, 3]);
        let b_shape = Shape::new_batched(2, 3, 2);
        
        let a = Tensor::<B>::new(
            Arc::clone(&ctx),
            a_shape.clone(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ).unwrap();
        
        let b = Tensor::<B>::new(
            Arc::clone(&ctx),
            b_shape.clone(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
              7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ).unwrap();

        println!("Attempting matmul between non-batched and batched tensors");
        visualize_shape_mismatch(a_shape.dims(), b_shape.dims());
        
        match a.matmul(&b) {
            Ok(_) => println!("❌ Error: Expected failure for batched/non-batched mismatch but got success"),
            Err(e) => {
                println!("✅ Successfully caught error: {}", e);
                println!("\nProblem: Cannot mix batched and non-batched tensors");
                println!("Solution: Either batch both tensors or use regular matmul");
            }
        }
    }

    println!("\nTest case 4: Invalid matrix dimensions in batch");
    {
        let a_shape = Shape::new_batched(2, 2, 3);
        let b_shape = Shape::new_batched(2, 4, 2);
        
        let a = Tensor::<B>::new(
            Arc::clone(&ctx),
            a_shape.clone(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
              7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ).unwrap();
        
        let b = Tensor::<B>::new(
            Arc::clone(&ctx),
            b_shape.clone(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
              9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        ).unwrap();

        println!("Attempting matmul with incompatible matrix dimensions");
        visualize_shape_mismatch(a_shape.dims(), b_shape.dims());
        
        match a.matmul(&b) {
            Ok(_) => println!("❌ Error: Expected failure for incompatible dimensions but got success"),
            Err(e) => {
                println!("✅ Successfully caught error: {}", e);
                println!("\nProblem: Inner matrix dimensions don't match (3 ≠ 4)");
                println!("Solution: Ensure the inner dimensions match (A: m×k, B: k×n)");
            }
        }
    }
}

fn main() {
    init_logging();
    
    test_batched_matmul::<CPUBackend>("CPU");
    test_batched_matmul::<MetalBackend>("GPU");
} 