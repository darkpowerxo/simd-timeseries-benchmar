//! SIMD Time Series Benchmark Suite
//!
//! A comprehensive benchmark suite comparing AVX2, AVX-512, and OpenCL (GPU)
//! implementations for time series deep learning operations.
//!
//! # Features
//!
//! - **SIMD Operations**: AVX2 and AVX-512 implementations for matrix operations,
//!   activation functions, attention mechanisms, and convolutions
//! - **GPU Computing**: OpenCL implementations for NVIDIA GPUs
//! - **Model Architectures**: MLP, Informer, TimesNet, and TFT implementations
//! - **Comprehensive Benchmarking**: Statistical benchmarking with Criterion.rs
//!
//! # Examples
//!
//! ```rust,no_run
//! use simd_timeseries_benchmark::{simd, ops};
//!
//! // Runtime feature detection
//! let features = simd::detect_features();
//! println!("AVX2 support: {}", features.avx2);
//! println!("AVX-512 support: {}", features.avx512f);
//! ```

#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

pub mod simd;
pub mod ops;

#[cfg(feature = "opencl")]
pub mod gpu;

pub mod models;
pub mod benchmarks;

pub use simd::Features;

/// Result type used throughout the library
pub type Result<T> = core::result::Result<T, Error>;

/// Error types for the benchmark suite
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// SIMD feature not available
    #[error("SIMD feature not available: {feature}")]
    SimdNotAvailable { 
        /// The specific SIMD feature that is not available
        feature: String 
    },
    
    /// Invalid matrix dimensions
    #[error("Invalid matrix dimensions: {message}")]
    InvalidDimensions { 
        /// Description of the dimension mismatch
        message: String 
    },
    
    /// OpenCL error
    #[cfg(feature = "opencl")]
    #[error("OpenCL error: {0}")]
    OpenCL(#[from] ocl::Error),
    
    /// General computation error
    #[error("Computation error: {message}")]
    Computation { 
        /// Description of the computation error
        message: String 
    },
}

/// Check if the current CPU supports the minimum required features
pub fn check_cpu_requirements() -> Features {
    simd::detect_features()
}
