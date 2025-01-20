use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ferroflow::compute::{CPUBackend, MetalBackend, ComputeBackend};
use ferroflow::tensor::{Tensor, Shape};
use std::sync::Arc;

fn bench_backend<B: ComputeBackend>(c: &mut Criterion, backend_name: &str) {
    let sizes = [1024, 10_240, 102_400, 1_024_000];
    let mut group = c.benchmark_group(format!("{} Operations", backend_name));

    // Benchmark different tensor sizes
    for size in sizes {
        let ctx = B::new().unwrap();
        let data: Vec<f32> = (0..size).map(|x| x as f32).collect();
        
        // Create tensors once to avoid init overhead
        let a = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new(vec![size]),
            &data
        ).unwrap();
        
        let tensor_b = Tensor::<B>::new(
            Arc::clone(&ctx),
            Shape::new(vec![size]),
            &data
        ).unwrap();

        // Element-wise addition
        group.bench_function(
            BenchmarkId::new("add", size), 
            |bencher| bencher.iter(|| {
                let result = black_box(&a).add(black_box(&tensor_b));
                black_box(result)
            })
        );
        
        // Element-wise multiplication
        group.bench_function(
            BenchmarkId::new("multiply", size),
            |bencher| bencher.iter(|| {
                let result = black_box(&a).multiply(black_box(&tensor_b));
                black_box(result)
            })
        );
        
        // Scalar multiplication
        group.bench_function(
            BenchmarkId::new("scalar_multiply", size),
            |bencher| bencher.iter(|| {
                let result = black_box(&a).scalar_multiply(black_box(2.0));
                black_box(result)
            })
        );

        // Memory transfer
        group.bench_function(
            BenchmarkId::new("read", size),
            |bencher| bencher.iter(|| {
                let result = black_box(&a).data();
                black_box(result)
            })
        );
    }

    group.finish();
}

fn bench_operations(c: &mut Criterion) {
    bench_backend::<CPUBackend>(c, "CPU");
    bench_backend::<MetalBackend>(c, "GPU");
}

criterion_group!(benches, bench_operations);
criterion_main!(benches); 