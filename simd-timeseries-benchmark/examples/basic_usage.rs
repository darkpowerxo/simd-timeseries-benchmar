use simd_timeseries_benchmark::{Result, ops, simd, models, benchmarks};

fn main() -> Result<()> {
    println!("AVX SIMD Time Series Benchmark Suite");
    println!("=====================================");
    
    // Feature detection
    let features = simd::detect_features();
    println!("\nDetected CPU Features:");
    println!("  AVX2: {}", features.avx2);
    println!("  AVX-512: {}", features.avx512f);
    
    // Matrix multiplication example
    println!("\n1. Matrix Multiplication Example");
    let m = 128;
    let n = 128; 
    let k = 128;
    
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    
    let start = std::time::Instant::now();
    ops::matmul_f32(&a, &b, &mut c, m, n, k)?;
    let duration = start.elapsed();
    
    println!("  {}x{}x{} matrix multiplication: {:?}", m, n, k, duration);
    println!("  Result sample: c[0] = {:.2}", c[0]);
    
    // Activation functions example
    println!("\n2. Activation Functions Example");
    let input = vec![0.5f32; 1024];
    let mut output = vec![0.0f32; 1024];
    
    let start = std::time::Instant::now();
    ops::relu(&input, &mut output)?;
    let relu_time = start.elapsed();
    
    let start = std::time::Instant::now();
    ops::gelu(&input, &mut output)?;
    let gelu_time = start.elapsed();
    
    println!("  ReLU (1024 elements): {:?}", relu_time);
    println!("  GELU (1024 elements): {:?}", gelu_time);
    
    // Time series models example
    println!("\n3. Time Series Models Example");
    let input_dim = 32;
    let hidden_dim = 64;
    
    let input_data = vec![0.1f32; input_dim * 10]; // Sample time series data
    
    let mlp = models::mlp_window::WindowedMLP::new(input_dim, &[hidden_dim]);
    
    let start = std::time::Instant::now();
    let prediction = mlp.predict(&input_data[0..input_dim])?;
    let model_time = start.elapsed();
    
    println!("  MLP windowing forward pass: {:?}", model_time);
    println!("  Prediction result: {:.4}", prediction);
    
    // Specialized workload examples
    println!("\n4. Specialized Workload Examples");
    
    // Docker workload
    let hash_data = vec![0x12345678u32; 1024];
    let mut hash_output = vec![0u32; 1024];
    
    let start = std::time::Instant::now();
    benchmarks::docker_example::hash_benchmark(&hash_data, &mut hash_output);
    let docker_time = start.elapsed();
    
    println!("  Docker hash benchmark: {:?}", docker_time);
    
    // AI workload (quantized)
    let a_i8 = vec![127i8; 256];
    let b_i8 = vec![64i8; 64];
    let mut output_i32 = vec![0i32; 4];
    
    let start = std::time::Instant::now();
    benchmarks::ai_workload::quantized_matmul_benchmark(&a_i8, &b_i8, &mut output_i32, 4, 64);
    let ai_time = start.elapsed();
    
    println!("  AI quantized matmul: {:?}", ai_time);
    
    // Finance workload
    let prices = vec![100.0f32; 252];
    let mut returns = vec![0.0f32; 251];
    
    let start = std::time::Instant::now();
    benchmarks::finance::log_returns(&prices, &mut returns);
    let finance_time = start.elapsed();
    
    println!("  Finance log returns: {:?}", finance_time);
    
    // Video encoding workload
    let block1 = vec![128u8; 256];
    let block2 = vec![127u8; 256];
    
    let start = std::time::Instant::now();
    let sad = benchmarks::video_encode::sad_16x16(&block1, &block2);
    let video_time = start.elapsed();
    
    println!("  Video SAD computation: {:?} (result: {})", video_time, sad);
    
    // Summary
    println!("\n5. Performance Summary");
    println!("  Features available: AVX2={}, AVX-512={}", features.avx2, features.avx512f);
    println!("  Total examples completed successfully!");
    
    Ok(())
}