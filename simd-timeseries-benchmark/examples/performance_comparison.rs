use simd_timeseries_benchmark::{Result, ops, simd};

fn main() -> Result<()> {
    println!("SIMD Performance Comparison");
    println!("===========================");
    
    let features = simd::detect_features();
    println!("CPU Features: AVX2={}, AVX-512={}", features.avx2, features.avx512f);
    
    // Matrix multiplication performance comparison
    println!("\nMatrix Multiplication Performance (1000x1000):");
    let size = 1000;
    let a = vec![1.0f32; size * size];
    let b = vec![1.0f32; size * size];
    let mut c = vec![0.0f32; size * size];
    
    // Scalar baseline
    let start = std::time::Instant::now();
    ops::matmul_f32_with_backend(&a, &b, &mut c, size, size, size, ops::MatMulBackend::Scalar)?;
    let scalar_time = start.elapsed();
    let scalar_gflops = (2.0 * size as f64 * size as f64 * size as f64) / (scalar_time.as_secs_f64() * 1e9);
    
    // Auto backend (best available)
    c.fill(0.0);
    let start = std::time::Instant::now();
    ops::matmul_f32(&a, &b, &mut c, size, size, size)?;
    let auto_time = start.elapsed();
    let auto_gflops = (2.0 * size as f64 * size as f64 * size as f64) / (auto_time.as_secs_f64() * 1e9);
    
    println!("  Scalar:      {:8.2} ms ({:.2} GFLOPS)", scalar_time.as_millis(), scalar_gflops);
    println!("  Auto (SIMD): {:8.2} ms ({:.2} GFLOPS)", auto_time.as_millis(), auto_gflops);
    println!("  Speedup:     {:.2}x", scalar_time.as_secs_f64() / auto_time.as_secs_f64());
    
    // Activation function comparison
    println!("\nActivation Function Performance (1M elements):");
    let size = 1_000_000;
    let input = vec![0.5f32; size];
    let mut output = vec![0.0f32; size];
    
    // ReLU
    let start = std::time::Instant::now();
    ops::relu(&input, &mut output)?;
    let relu_time = start.elapsed();
    let relu_throughput = size as f64 / (relu_time.as_secs_f64() * 1e9);
    
    // GELU
    let start = std::time::Instant::now();
    ops::gelu(&input, &mut output)?;
    let gelu_time = start.elapsed();
    let gelu_throughput = size as f64 / (gelu_time.as_secs_f64() * 1e9);
    
    // Tanh
    let start = std::time::Instant::now();
    ops::tanh(&input, &mut output)?;
    let tanh_time = start.elapsed();
    let tanh_throughput = size as f64 / (tanh_time.as_secs_f64() * 1e9);
    
    println!("  ReLU: {:8.2} ms ({:.2} G elements/sec)", relu_time.as_millis(), relu_throughput);
    println!("  GELU: {:8.2} ms ({:.2} G elements/sec)", gelu_time.as_millis(), gelu_throughput);
    println!("  Tanh: {:8.2} ms ({:.2} G elements/sec)", tanh_time.as_millis(), tanh_throughput);
    
    // Memory bandwidth test
    println!("\nMemory Bandwidth Test (10M floats):");
    let size = 10_000_000;
    let src = vec![1.0f32; size];
    let mut dst = vec![0.0f32; size];
    
    let start = std::time::Instant::now();
    // Simple copy operation
    for i in 0..size {
        dst[i] = src[i] * 2.0;
    }
    let copy_time = start.elapsed();
    let bandwidth_gb_s = (size * 2 * 4) as f64 / (copy_time.as_secs_f64() * 1e9); // 2 arrays * 4 bytes
    
    println!("  Copy+Scale: {:8.2} ms ({:.2} GB/s)", copy_time.as_millis(), bandwidth_gb_s);
    
    // Feature detection overhead
    println!("\nFeature Detection Overhead:");
    let iterations = 1000;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _features = simd::detect_features();
    }
    let detection_time = start.elapsed();
    
    println!("  {} detections in {:?} ({:.2} ns per detection)", 
             iterations, detection_time, detection_time.as_nanos() as f64 / iterations as f64);
    
    Ok(())
}