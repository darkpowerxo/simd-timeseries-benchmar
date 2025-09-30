//! Matrix multiplication operations
//!
//! High-performance matrix multiplication implementations using different SIMD backends.

use crate::{Result, Error};
use crate::simd::detect_features;

/// Matrix multiplication dispatch based on available CPU features
pub fn matmul_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(Error::InvalidDimensions {
            message: format!("Matrix dimensions mismatch: A{}x{}, B{}x{}, C{}x{}", 
                           m, k, k, n, m, n)
        });
    }

    let features = detect_features();
    
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if features.avx512f {
            unsafe {
                crate::simd::avx512::matmul_f32_avx512(a, b, c, m, n, k);
            }
            return Ok(());
        }
        
        if features.avx2 {
            unsafe {
                crate::simd::avx2::matmul_f32_avx2(a, b, c, m, n, k);
            }
            return Ok(());
        }
    }
    
    // Fallback to scalar implementation
    crate::simd::fallback::matmul_f32(a, b, c, m, n, k);
    Ok(())
}

/// Optimized matrix multiplication with explicit backend selection
pub enum MatMulBackend {
    /// Automatically select best available backend
    Auto,
    /// Force scalar fallback
    Scalar,
    /// Force AVX2 (will error if not available)
    Avx2,
    /// Force AVX-512 (will error if not available)
    Avx512,
}

/// Matrix multiplication with explicit backend selection
pub fn matmul_f32_with_backend(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    backend: MatMulBackend,
) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(Error::InvalidDimensions {
            message: format!("Matrix dimensions mismatch: A{}x{}, B{}x{}, C{}x{}", 
                           m, k, k, n, m, n)
        });
    }

    match backend {
        MatMulBackend::Auto => matmul_f32(a, b, c, m, n, k),
        MatMulBackend::Scalar => {
            crate::simd::fallback::matmul_f32(a, b, c, m, n, k);
            Ok(())
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        MatMulBackend::Avx2 => {
            if !detect_features().avx2 {
                return Err(Error::SimdNotAvailable { 
                    feature: "AVX2".to_string() 
                });
            }
            unsafe {
                crate::simd::avx2::matmul_f32_avx2(a, b, c, m, n, k);
            }
            Ok(())
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        MatMulBackend::Avx512 => {
            if !detect_features().avx512f {
                return Err(Error::SimdNotAvailable { 
                    feature: "AVX-512".to_string() 
                });
            }
            unsafe {
                crate::simd::avx512::matmul_f32_avx512(a, b, c, m, n, k);
            }
            Ok(())
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        MatMulBackend::Avx2 | MatMulBackend::Avx512 => {
            Err(Error::SimdNotAvailable { 
                feature: "x86 SIMD not available on this platform".to_string() 
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_matmul_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = [5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
        let mut c = [0.0; 4];
        
        matmul_f32(&a, &b, &mut c, 2, 2, 2).unwrap();
        
        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        let expected = [19.0, 22.0, 43.0, 50.0];
        
        for (actual, exp) in c.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, exp, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_matmul_all_backends() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        
        let mut c_scalar = [0.0; 4];
        let mut c_auto = [0.0; 4];
        
        matmul_f32_with_backend(&a, &b, &mut c_scalar, 2, 2, 2, MatMulBackend::Scalar).unwrap();
        matmul_f32_with_backend(&a, &b, &mut c_auto, 2, 2, 2, MatMulBackend::Auto).unwrap();
        
        for (scalar, auto) in c_scalar.iter().zip(c_auto.iter()) {
            assert_relative_eq!(scalar, auto, epsilon = 1e-6);
        }
    }
}