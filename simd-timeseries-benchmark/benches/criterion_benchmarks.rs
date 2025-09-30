use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use simd_timeseries_benchmark::{ops, simd, benchmarks};

fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    
    // Test different matrix sizes
    let sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)];
    
    for (m, n, k) in sizes.iter() {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c_out = vec![0.0f32; m * n];
        
        group.bench_with_input(
            BenchmarkId::new("auto", format!("{}x{}x{}", m, n, k)),
            &(*m, *n, *k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    ops::matmul_f32(
                        black_box(&a),
                        black_box(&b), 
                        black_box(&mut c_out),
                        m, n, k
                    ).unwrap();
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}x{}", m, n, k)),
            &(*m, *n, *k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    ops::matmul_f32_with_backend(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c_out), 
                        m, n, k,
                        ops::MatMulBackend::Scalar
                    ).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");
    
    let sizes = [1024, 4096, 16384];
    
    for size in sizes.iter() {
        let input = vec![1.0f32; *size];
        let mut output = vec![0.0f32; *size];
        
        group.bench_with_input(
            BenchmarkId::new("relu", size),
            size,
            |bench, &_size| {
                bench.iter(|| {
                    ops::relu(black_box(&input), black_box(&mut output)).unwrap();
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("gelu", size),
            size,
            |bench, &_size| {
                bench.iter(|| {
                    ops::gelu(black_box(&input), black_box(&mut output)).unwrap();
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("tanh", size),
            size,
            |bench, &_size| {
                bench.iter(|| {
                    ops::tanh(black_box(&input), black_box(&mut output)).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

fn bench_feature_detection(c: &mut Criterion) {
    c.bench_function("feature_detection", |b| {
        b.iter(|| {
            let features = simd::detect_features();
            black_box(features);
        });
    });
}

fn bench_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("workloads");
    
    // Docker workload
    let hash_data = vec![0x12345678u32; 1024];
    let mut hash_output = vec![0u32; 1024];
    
    group.bench_function("docker_hash", |b| {
        b.iter(|| {
            benchmarks::docker_example::hash_benchmark(
                black_box(&hash_data),
                black_box(&mut hash_output)
            );
        });
    });
    
    // AI workload  
    let a_i8 = vec![127i8; 256];
    let b_i8 = vec![64i8; 64];
    let mut output_i32 = vec![0i32; 4];
    
    group.bench_function("ai_quantized_matmul", |b| {
        b.iter(|| {
            benchmarks::ai_workload::quantized_matmul_benchmark(
                black_box(&a_i8),
                black_box(&b_i8),
                black_box(&mut output_i32),
                4, 64
            );
        });
    });
    
    // Finance workload
    let prices = vec![100.0f32; 252];
    let mut returns = vec![0.0f32; 251];
    
    group.bench_function("finance_log_returns", |b| {
        b.iter(|| {
            benchmarks::finance::log_returns(
                black_box(&prices),
                black_box(&mut returns)
            );
        });
    });
    
    // Video workload
    let block1 = vec![128u8; 256];
    let block2 = vec![127u8; 256];
    
    group.bench_function("video_sad", |b| {
        b.iter(|| {
            let sad = benchmarks::video_encode::sad_16x16(
                black_box(&block1),
                black_box(&block2)
            );
            black_box(sad);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_feature_detection,
    bench_matrix_multiplication,
    bench_activations,
    bench_workloads
);
criterion_main!(benches);