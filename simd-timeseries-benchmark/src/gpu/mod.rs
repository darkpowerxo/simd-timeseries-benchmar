//! GPU implementations using OpenCL
//!
//! This module provides OpenCL implementations for NVIDIA GPUs.

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(feature = "opencl")]
pub use opencl::*;