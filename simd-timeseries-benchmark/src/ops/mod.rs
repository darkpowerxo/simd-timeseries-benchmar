//! Core operations module
//!
//! This module provides the main mathematical operations used in time series
//! deep learning, with implementations across different SIMD backends.

pub mod matmul;
pub mod activation;
pub mod attention;
pub mod conv1d;

pub use self::{
    matmul::*,
    activation::*,
    attention::*,
    conv1d::*,
};