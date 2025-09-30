//! SIMD implementations and feature detection
//! 
//! This module provides runtime CPU feature detection and safe SIMD implementations
//! with automatic fallback to scalar code when SIMD instructions are not available.
//!
//! # Safety
//! All unsafe SIMD code is isolated in this module and properly documented.
//! Runtime feature detection ensures that only supported instructions are used.
//!
//! # Architecture
//! - `detect_features()`: Runtime CPU feature detection
//! - `dispatch_*()`: Safe function dispatch based on available features
//! - `Features`: Comprehensive CPU feature detection results
//! - Automatic fallback to scalar implementations when SIMD is unavailable

use cfg_if::cfg_if;
use core::sync::atomic::{AtomicBool, Ordering};

pub mod fallback;

cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        pub mod avx2;
        pub mod avx512;
        
        /// CPU feature detection results with comprehensive x86 SIMD support
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct Features {
            /// AVX2 support (256-bit vectors)
            pub avx2: bool,
            /// FMA (Fused Multiply-Add) support
            pub fma: bool,
            /// AVX-512 Foundation (F) support (512-bit vectors, basic ops)
            pub avx512f: bool,
            /// AVX-512 Vector Length (VL) support (128/256-bit vectors with AVX-512)
            pub avx512vl: bool,
            /// AVX-512 Byte and Word (BW) support (8/16-bit integer ops)
            pub avx512bw: bool,
            /// AVX-512 Doubleword and Quadword (DQ) support (32/64-bit integer ops)
            pub avx512dq: bool,
            /// AVX-512 Conflict Detection (CD) support (duplicate detection)
            pub avx512cd: bool,
            /// AVX-512 Exponential and Reciprocal (ER) support (transcendental functions)
            pub avx512er: bool,
            /// AVX-512 Vector Neural Network Instructions (VNNI) support (INT8 dot products)
            pub avx512vnni: bool,
            /// AVX-512 BFloat16 support (16-bit brain floating point)
            pub avx512bf16: bool,
            /// AVX-512 Vector Byte Manipulation Instructions (VBMI) support
            pub avx512vbmi: bool,
            /// AVX-512 Integer Fused Multiply Add (IFMA) support
            pub avx512ifma: bool,
        }
        
        /// Cached feature detection to avoid repeated CPUID calls
        static FEATURES_DETECTED: AtomicBool = AtomicBool::new(false);
        static mut CACHED_FEATURES: Features = Features {
            avx2: false,
            fma: false,
            avx512f: false,
            avx512vl: false,
            avx512bw: false,
            avx512dq: false,
            avx512cd: false,
            avx512er: false,
            avx512vnni: false,
            avx512bf16: false,
            avx512vbmi: false,
            avx512ifma: false,
        };
        
        /// Detect available CPU features at runtime with caching
        /// 
        /// This function performs CPUID-based feature detection only once and caches
        /// the results for subsequent calls to avoid performance overhead.
        /// 
        /// # Safety
        /// Uses atomic operations to ensure thread-safe caching across multiple threads.
        pub fn detect_features() -> Features {
            // Check if we've already detected features
            if FEATURES_DETECTED.load(Ordering::Acquire) {
                // SAFETY: CACHED_FEATURES is only written once before FEATURES_DETECTED is set to true
                return unsafe { CACHED_FEATURES };
            }
            
            // Perform feature detection
            let features = Features {
                avx2: is_x86_feature_detected!("avx2"),
                fma: is_x86_feature_detected!("fma"),
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512vl: is_x86_feature_detected!("avx512vl"),
                avx512bw: is_x86_feature_detected!("avx512bw"),
                avx512dq: is_x86_feature_detected!("avx512dq"),
                avx512cd: is_x86_feature_detected!("avx512cd"),
                avx512er: is_x86_feature_detected!("avx512er"),
                avx512vnni: is_x86_feature_detected!("avx512vnni"),
                avx512bf16: is_x86_feature_detected!("avx512bf16"),
                avx512vbmi: is_x86_feature_detected!("avx512vbmi"),
                avx512ifma: is_x86_feature_detected!("avx512ifma"),
            };
            
            // Cache the results
            // SAFETY: This is safe because we only write once before setting the atomic flag
            unsafe {
                CACHED_FEATURES = features;
            }
            FEATURES_DETECTED.store(true, Ordering::Release);
            
            features
        }
        
        impl Features {
            /// Check if basic AVX-512 support is available (F + VL)
            pub fn has_avx512_basic(&self) -> bool {
                self.avx512f && self.avx512vl
            }
            
            /// Check if AVX-512 with enhanced integer support is available
            pub fn has_avx512_enhanced(&self) -> bool {
                self.avx512f && self.avx512vl && self.avx512bw && self.avx512dq
            }
            
            /// Check if AVX-512 VNNI (neural network) instructions are available
            pub fn has_avx512_vnni(&self) -> bool {
                self.has_avx512_basic() && self.avx512vnni
            }
            
            /// Check if AVX-512 BF16 (brain float) instructions are available
            pub fn has_avx512_bf16(&self) -> bool {
                self.has_avx512_basic() && self.avx512bf16
            }
            
            /// Get the best available SIMD level for the current CPU
            pub fn best_simd_level(&self) -> SimdLevel {
                if self.has_avx512_enhanced() {
                    SimdLevel::Avx512
                } else if self.avx2 {
                    SimdLevel::Avx2
                } else {
                    SimdLevel::Scalar
                }
            }
        }
        
        /// SIMD implementation levels available on this platform
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub enum SimdLevel {
            /// Scalar (no SIMD) fallback
            Scalar = 0,
            /// AVX2 (256-bit vectors)
            Avx2 = 1,
            /// AVX-512 (512-bit vectors)
            Avx512 = 2,
        }
        
        impl core::fmt::Display for SimdLevel {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                match self {
                    SimdLevel::Scalar => write!(f, "Scalar"),
                    SimdLevel::Avx2 => write!(f, "AVX2"),
                    SimdLevel::Avx512 => write!(f, "AVX-512"),
                }
            }
        }
        
        /// Safe dispatch for matrix multiplication operations
        /// 
        /// Automatically selects the best available implementation based on CPU features.
        /// 
        /// # Arguments
        /// * `a` - Left matrix (m×k)
        /// * `b` - Right matrix (k×n) 
        /// * `c` - Output matrix (m×n)
        /// * `m`, `n`, `k` - Matrix dimensions
        /// 
        /// # Safety
        /// This function is safe to call as it performs runtime feature detection
        /// and only calls SIMD implementations when the required features are available.
        pub fn dispatch_matmul_f32(
            a: &[f32],
            b: &[f32],
            c: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
        ) -> crate::Result<()> {
            let features = detect_features();
            
            match features.best_simd_level() {
                SimdLevel::Avx512 if features.has_avx512_enhanced() => {
                    // SAFETY: We've verified AVX-512 support via runtime detection
                    unsafe { avx512::matmul_f32_avx512(a, b, c, m, n, k) }
                },
                SimdLevel::Avx2 if features.avx2 => {
                    // SAFETY: We've verified AVX2 support via runtime detection
                    unsafe { avx2::matmul_f32_avx2(a, b, c, m, n, k) }
                },
                _ => {
                    fallback::matmul_f32(a, b, c, m, n, k)?
                }
            }
            Ok(())
        }
        
        /// Safe dispatch for ReLU activation function
        /// 
        /// # Arguments
        /// * `input` - Input array
        /// * `output` - Output array (same length as input)
        /// 
        /// # Safety
        /// This function is safe to call as it performs runtime feature detection.
        pub fn dispatch_relu_f32(input: &[f32], output: &mut [f32]) -> crate::Result<()> {
            if input.len() != output.len() {
                return Err(crate::Error::InvalidDimensions {
                    message: format!("Input and output arrays must have the same length: {} vs {}", 
                                   input.len(), output.len())
                });
            }
            
            let features = detect_features();
            
            match features.best_simd_level() {
                SimdLevel::Avx512 if features.has_avx512_basic() => {
                    // SAFETY: We've verified AVX-512 support via runtime detection
                    unsafe { avx512::relu_f32_avx512(input, output) }
                },
                SimdLevel::Avx2 if features.avx2 => {
                    // SAFETY: We've verified AVX2 support via runtime detection
                    unsafe { avx2::relu_f32_avx2(input, output) }
                },
                _ => {
                    fallback::relu_f32(input, output)?
                }
            }
            Ok(())
        }
        
        /// Safe dispatch for GELU activation function
        /// 
        /// # Arguments
        /// * `input` - Input array
        /// * `output` - Output array (same length as input)
        pub fn dispatch_gelu_f32(input: &[f32], output: &mut [f32]) -> crate::Result<()> {
            if input.len() != output.len() {
                return Err(crate::Error::InvalidDimensions {
                    message: format!("Input and output arrays must have the same length: {} vs {}", 
                                   input.len(), output.len())
                });
            }
            
            let features = detect_features();
            
            match features.best_simd_level() {
                SimdLevel::Avx512 if features.has_avx512_basic() => {
                    // SAFETY: We've verified AVX-512 support via runtime detection
                    unsafe { avx512::gelu_f32_avx512(input, output) }
                },
                SimdLevel::Avx2 if features.avx2 => {
                    // SAFETY: We've verified AVX2 support via runtime detection
                    unsafe { avx2::gelu_f32_avx2(input, output) }
                },
                _ => {
                    fallback::gelu_f32(input, output)?
                }
            }
            Ok(())
        }
        
    } else {
        /// CPU feature detection results (non-x86 platforms)
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct Features {
            /// AVX2 support (always false on non-x86)
            pub avx2: bool,
            /// FMA support (always false on non-x86)
            pub fma: bool,
            /// AVX-512 Foundation (F) support (always false on non-x86)
            pub avx512f: bool,
            /// AVX-512 Vector Length (VL) support (always false on non-x86)
            pub avx512vl: bool,
            /// AVX-512 Byte and Word (BW) support (always false on non-x86)
            pub avx512bw: bool,
            /// AVX-512 Doubleword and Quadword (DQ) support (always false on non-x86)
            pub avx512dq: bool,
            /// AVX-512 Conflict Detection (CD) support (always false on non-x86)
            pub avx512cd: bool,
            /// AVX-512 Exponential and Reciprocal (ER) support (always false on non-x86)
            pub avx512er: bool,
            /// AVX-512 Vector Neural Network Instructions (VNNI) support (always false on non-x86)
            pub avx512vnni: bool,
            /// AVX-512 BFloat16 support (always false on non-x86)
            pub avx512bf16: bool,
            /// AVX-512 Vector Byte Manipulation Instructions (VBMI) support (always false on non-x86)
            pub avx512vbmi: bool,
            /// AVX-512 Integer Fused Multiply Add (IFMA) support (always false on non-x86)
            pub avx512ifma: bool,
        }
        
        impl Features {
            /// Check if basic AVX-512 support is available (always false on non-x86)
            pub fn has_avx512_basic(&self) -> bool { false }
            
            /// Check if AVX-512 with enhanced integer support is available (always false on non-x86)
            pub fn has_avx512_enhanced(&self) -> bool { false }
            
            /// Check if AVX-512 VNNI instructions are available (always false on non-x86)
            pub fn has_avx512_vnni(&self) -> bool { false }
            
            /// Check if AVX-512 BF16 instructions are available (always false on non-x86)
            pub fn has_avx512_bf16(&self) -> bool { false }
            
            /// Get the best available SIMD level (always Scalar on non-x86)
            pub fn best_simd_level(&self) -> SimdLevel { SimdLevel::Scalar }
        }
        
        /// SIMD implementation levels (only Scalar available on non-x86)
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub enum SimdLevel {
            /// Scalar (no SIMD) fallback
            Scalar = 0,
        }
        
        impl core::fmt::Display for SimdLevel {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "Scalar")
            }
        }
        
        /// Detect available CPU features at runtime (non-x86 platforms)
        pub fn detect_features() -> Features {
            Features {
                avx2: false,
                fma: false,
                avx512f: false,
                avx512vl: false,
                avx512bw: false,
                avx512dq: false,
                avx512cd: false,
                avx512er: false,
                avx512vnni: false,
                avx512bf16: false,
                avx512vbmi: false,
                avx512ifma: false,
            }
        }
        
        /// Safe dispatch for matrix multiplication (scalar only on non-x86)
        pub fn dispatch_matmul_f32(
            a: &[f32],
            b: &[f32],
            c: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
        ) -> crate::Result<()> {
            fallback::matmul_f32(a, b, c, m, n, k)
        }
        
        /// Safe dispatch for ReLU activation (scalar only on non-x86)
        pub fn dispatch_relu_f32(input: &[f32], output: &mut [f32]) -> crate::Result<()> {
            if input.len() != output.len() {
                return Err(crate::Error::InvalidDimensions {
                    message: format!("Input and output arrays must have the same length: {} vs {}", 
                                   input.len(), output.len())
                });
            }
            fallback::relu_f32(input, output)
        }
        
        /// Safe dispatch for GELU activation (scalar only on non-x86)
        pub fn dispatch_gelu_f32(input: &[f32], output: &mut [f32]) -> crate::Result<()> {
            if input.len() != output.len() {
                return Err(crate::Error::InvalidDimensions {
                    message: format!("Input and output arrays must have the same length: {} vs {}", 
                                   input.len(), output.len())
                });
            }
            fallback::gelu_f32(input, output)
        }
    }
}

/// Print CPU feature detection results for debugging
pub fn print_cpu_features() {
    let features = detect_features();
    println!("CPU Feature Detection Results:");
    println!("  Best SIMD Level: {}", features.best_simd_level());
    
    cfg_if! {
        if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
            println!("  AVX2:           {}", features.avx2);
            println!("  FMA:            {}", features.fma);
            println!("  AVX-512 F:      {}", features.avx512f);
            println!("  AVX-512 VL:     {}", features.avx512vl);
            println!("  AVX-512 BW:     {}", features.avx512bw);
            println!("  AVX-512 DQ:     {}", features.avx512dq);
            println!("  AVX-512 CD:     {}", features.avx512cd);
            println!("  AVX-512 ER:     {}", features.avx512er);
            println!("  AVX-512 VNNI:   {}", features.avx512vnni);
            println!("  AVX-512 BF16:   {}", features.avx512bf16);
            println!("  AVX-512 VBMI:   {}", features.avx512vbmi);
            println!("  AVX-512 IFMA:   {}", features.avx512ifma);
            println!();
            println!("Feature Combinations:");
            println!("  AVX-512 Basic:     {}", features.has_avx512_basic());
            println!("  AVX-512 Enhanced:  {}", features.has_avx512_enhanced());
            println!("  AVX-512 VNNI:      {}", features.has_avx512_vnni());
            println!("  AVX-512 BF16:      {}", features.has_avx512_bf16());
        } else {
            println!("  Platform:       Non-x86 (SIMD not supported)");
        }
    }
}

/// Check CPU requirements for different workload types
/// 
/// This function provides recommendations based on detected CPU features
/// for optimal performance in different scenarios.
pub fn check_cpu_requirements() -> CpuRequirementsReport {
    let features = detect_features();
    
    CpuRequirementsReport {
        features,
        recommendations: generate_recommendations(&features),
        workload_support: check_workload_support(&features),
    }
}

/// CPU requirements and recommendations report
#[derive(Debug, Clone)]
pub struct CpuRequirementsReport {
    /// Detected CPU features
    pub features: Features,
    /// Performance recommendations
    pub recommendations: Vec<String>,
    /// Workload-specific support information
    pub workload_support: WorkloadSupport,
}

/// Workload-specific CPU support information
#[derive(Debug, Clone)]
pub struct WorkloadSupport {
    /// General deep learning operations support level
    pub deep_learning: SupportLevel,
    /// Docker workload (conflict detection) support
    pub docker: SupportLevel,
    /// AI workload (quantized inference) support  
    pub ai_quantized: SupportLevel,
    /// Finance workload (high precision) support
    pub finance: SupportLevel,
    /// Video encoding workload support
    pub video_encoding: SupportLevel,
}

/// Support level for different workloads
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportLevel {
    /// Not supported (will use scalar fallback)
    NotSupported,
    /// Basic support (AVX2 or basic AVX-512)
    Basic,
    /// Optimized support (enhanced AVX-512 features)
    Optimized,
    /// Specialized support (workload-specific instructions)
    Specialized,
}

impl core::fmt::Display for SupportLevel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SupportLevel::NotSupported => write!(f, "Not Supported"),
            SupportLevel::Basic => write!(f, "Basic"),
            SupportLevel::Optimized => write!(f, "Optimized"),
            SupportLevel::Specialized => write!(f, "Specialized"),
        }
    }
}

fn generate_recommendations(features: &Features) -> Vec<String> {
    let mut recommendations = Vec::new();
    
    if !features.avx2 {
        recommendations.push("❌ AVX2 not detected - all operations will use scalar fallback. Consider upgrading to a newer CPU (Intel Haswell/AMD Excavator or later).".to_string());
    } else {
        recommendations.push("✅ AVX2 detected - good performance expected for most operations.".to_string());
    }
    
    if features.has_avx512_basic() {
        recommendations.push("✅ AVX-512 basic support detected - excellent performance expected.".to_string());
        
        if features.has_avx512_enhanced() {
            recommendations.push("✅ AVX-512 enhanced support detected - optimal performance for all operations.".to_string());
        } else {
            recommendations.push("⚠️  AVX-512 BW/DQ not detected - some integer operations may be slower.".to_string());
        }
        
        if features.has_avx512_vnni() {
            recommendations.push("✅ AVX-512 VNNI detected - excellent performance for quantized neural networks.".to_string());
        }
        
        if features.has_avx512_bf16() {
            recommendations.push("✅ AVX-512 BF16 detected - optimal performance for mixed-precision training.".to_string());
        }
        
        if features.avx512cd {
            recommendations.push("✅ AVX-512 CD detected - specialized support for Docker/container workloads.".to_string());
        }
        
        if features.avx512er {
            recommendations.push("✅ AVX-512 ER detected - excellent performance for financial/scientific computations.".to_string());
        }
    } else if !features.avx2 {
        recommendations.push("❌ No SIMD support detected - performance will be significantly limited.".to_string());
    }
    
    if !features.fma && features.avx2 {
        recommendations.push("⚠️  FMA not detected - matrix operations may be slower than optimal.".to_string());
    }
    
    recommendations
}

fn check_workload_support(features: &Features) -> WorkloadSupport {
    WorkloadSupport {
        deep_learning: if features.has_avx512_enhanced() {
            SupportLevel::Optimized
        } else if features.avx2 {
            SupportLevel::Basic
        } else {
            SupportLevel::NotSupported
        },
        
        docker: if features.avx512cd && features.has_avx512_basic() {
            SupportLevel::Specialized
        } else if features.has_avx512_basic() {
            SupportLevel::Optimized
        } else if features.avx2 {
            SupportLevel::Basic
        } else {
            SupportLevel::NotSupported
        },
        
        ai_quantized: if features.has_avx512_vnni() {
            SupportLevel::Specialized
        } else if features.has_avx512_enhanced() {
            SupportLevel::Optimized
        } else if features.avx2 {
            SupportLevel::Basic
        } else {
            SupportLevel::NotSupported
        },
        
        finance: if features.avx512er && features.has_avx512_bf16() {
            SupportLevel::Specialized
        } else if features.has_avx512_basic() {
            SupportLevel::Optimized
        } else if features.avx2 {
            SupportLevel::Basic
        } else {
            SupportLevel::NotSupported
        },
        
        video_encoding: if features.has_avx512_enhanced() {
            SupportLevel::Optimized
        } else if features.avx2 {
            SupportLevel::Basic
        } else {
            SupportLevel::NotSupported
        },
    }
}

impl CpuRequirementsReport {
    /// Print a comprehensive CPU requirements report
    pub fn print_report(&self) {
        println!("\n🖥️  CPU Requirements Report");
        println!("═══════════════════════════════════════");
        
        println!("\n📊 Detected Features:");
        println!("  Best SIMD Level: {}", self.features.best_simd_level());
        
        println!("\n💡 Recommendations:");
        for rec in &self.recommendations {
            println!("  {}", rec);
        }
        
        println!("\n🎯 Workload Support:");
        println!("  Deep Learning:     {}", self.workload_support.deep_learning);
        println!("  Docker/Containers: {}", self.workload_support.docker);
        println!("  AI Quantized:      {}", self.workload_support.ai_quantized);
        println!("  Finance/Scientific:{}", self.workload_support.finance);
        println!("  Video Encoding:    {}", self.workload_support.video_encoding);
        
        println!("\nℹ️  For detailed feature breakdown, use print_cpu_features()");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_feature_detection() {
        let features = detect_features();
        
        // Feature detection should be consistent
        let features2 = detect_features();
        assert_eq!(features.avx2, features2.avx2);
        assert_eq!(features.avx512f, features2.avx512f);
        
        // Basic sanity checks
        println!("Detected features: {:?}", features);
        
        // Test feature combinations
        if features.avx512f {
            // If AVX-512F is available, check logical combinations
            println!("AVX-512 basic support: {}", features.has_avx512_basic());
            println!("AVX-512 enhanced support: {}", features.has_avx512_enhanced());
        }
        
        if features.avx2 {
            println!("AVX2 support detected");
        }
        
        // Best SIMD level should be deterministic
        let level1 = features.best_simd_level();
        let level2 = features.best_simd_level();
        assert_eq!(level1, level2);
    }
    
    #[test]
    fn test_dispatch_functions() {
        // Test matrix multiplication dispatch
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        
        dispatch_matmul_f32(&a, &b, &mut c, 2, 2, 2).unwrap();
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
        
        // Test ReLU dispatch
        let input = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let mut output = [0.0f32; 5];
        
        dispatch_relu_f32(&input, &mut output).unwrap();
        assert_eq!(output, [0.0, 0.0, 0.0, 1.0, 2.0]);
        
        // Test GELU dispatch
        let input = [0.0f32, 1.0, -1.0];
        let mut output = [0.0f32; 3];
        
        dispatch_gelu_f32(&input, &mut output).unwrap();
        assert_relative_eq!(output[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 0.8413447, epsilon = 1e-4); // Slightly more tolerant
        assert_relative_eq!(output[2], -0.15865526, epsilon = 1e-4);
    }
    
    #[test]
    fn test_error_cases() {
        // Test dimension mismatch errors
        let input = [1.0f32, 2.0, 3.0];
        let mut wrong_output = [0.0f32; 2];
        
        let result = dispatch_relu_f32(&input, &mut wrong_output);
        assert!(result.is_err());
        
        let result = dispatch_gelu_f32(&input, &mut wrong_output);
        assert!(result.is_err());
        
        // Test invalid matrix dimensions
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let mut c = [0.0f32; 1];
        
        let result = dispatch_matmul_f32(&a, &b, &mut c, 2, 2, 2);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_simd_levels() {
        use core::cmp::Ordering;
        
        // Test ordering of SIMD levels
        assert_eq!(SimdLevel::Scalar.cmp(&SimdLevel::Avx2), Ordering::Less);
        
        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                assert_eq!(SimdLevel::Avx2.cmp(&SimdLevel::Avx512), Ordering::Less);
                
                // Test display formatting
                assert_eq!(format!("{}", SimdLevel::Scalar), "Scalar");
                assert_eq!(format!("{}", SimdLevel::Avx2), "AVX2");
                assert_eq!(format!("{}", SimdLevel::Avx512), "AVX-512");
            } else {
                // On non-x86, only Scalar is available
                assert_eq!(format!("{}", SimdLevel::Scalar), "Scalar");
            }
        }
    }
    
    /// Test that demonstrates how to use the feature detection system
    #[test]
    fn test_usage_example() {
        // Print CPU capabilities for debugging
        print_cpu_features();
        
        // Example of how to check for specific features
        let features = detect_features();
        
        if features.avx2 {
            println!("AVX2 is available - can use 256-bit vectors");
        }
        
        if features.has_avx512_basic() {
            println!("AVX-512 basic is available - can use 512-bit vectors");
        }
        
        if features.has_avx512_vnni() {
            println!("AVX-512 VNNI is available - can use INT8 neural network instructions");
        }
        
        // The dispatch functions automatically select the best implementation
        let input = vec![1.0f32; 1000];
        let mut output = vec![0.0f32; 1000];
        
        // This will automatically use the best available SIMD implementation
        dispatch_relu_f32(&input, &mut output).unwrap();
        
        // Verify the result
        for &val in &output {
            assert_eq!(val, 1.0f32);
        }
    }
}