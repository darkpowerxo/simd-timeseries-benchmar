//! AVX-512 SIMD implementations
//!
//! This module provides AVX-512-optimized implementations of core operations
//! using 16-wide FP32 SIMD instructions.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::*;

/// AVX-512 matrix multiplication with tiling
#[target_feature(enable = "avx512f")]
pub unsafe fn matmul_f32_avx512(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    
    const TILE_SIZE: usize = 128;
    const SIMD_WIDTH: usize = 16;
    
    // Tiled matrix multiplication for cache efficiency
    for i_tile in (0..m).step_by(TILE_SIZE) {
        for j_tile in (0..n).step_by(TILE_SIZE) {
            for k_tile in (0..k).step_by(TILE_SIZE) {
                let i_end = (i_tile + TILE_SIZE).min(m);
                let j_end = (j_tile + TILE_SIZE).min(n);
                let k_end = (k_tile + TILE_SIZE).min(k);
                
                for i in i_tile..i_end {
                    let mut j = j_tile;
                    
                    // Process 16 elements at a time using AVX-512
                    while j + SIMD_WIDTH <= j_end {
                        let mut acc = _mm512_setzero_ps();
                        
                        for l in k_tile..k_end {
                            let a_val = _mm512_set1_ps(a[i * k + l]);
                            let b_ptr = unsafe { b.as_ptr().add(l * n + j) };
                            let b_val = unsafe { _mm512_loadu_ps(b_ptr) };
                            acc = _mm512_fmadd_ps(a_val, b_val, acc);
                        }
                        
                        let c_ptr = unsafe { c.as_mut_ptr().add(i * n + j) };
                        let c_val = unsafe { _mm512_loadu_ps(c_ptr) };
                        let result = _mm512_add_ps(c_val, acc);
                        unsafe { _mm512_storeu_ps(c_ptr, result) };
                        
                        j += SIMD_WIDTH;
                    }
                    
                    // Handle remaining elements
                    for j in j..j_end {
                        let mut sum = c[i * n + j];
                        for l in k_tile..k_end {
                            sum += a[i * k + l] * b[l * n + j];
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

/// AVX-512 ReLU activation
#[target_feature(enable = "avx512f")]
pub unsafe fn relu_f32_avx512(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    const SIMD_WIDTH: usize = 16;
    let len = input.len();
    let mut i = 0;
    
    let zero = _mm512_setzero_ps();
    
    // Process 16 elements at a time
    while i + SIMD_WIDTH <= len {
        let inp = unsafe { _mm512_loadu_ps(input.as_ptr().add(i)) };
        let result = _mm512_max_ps(inp, zero);
        unsafe { _mm512_storeu_ps(output.as_mut_ptr().add(i), result) };
        i += SIMD_WIDTH;
    }
    
    // Handle remaining elements
    for j in i..len {
        output[j] = input[j].max(0.0);
    }
}

/// AVX-512 tanh activation (standard implementation)
#[target_feature(enable = "avx512f")]
pub unsafe fn tanh_f32_avx512(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    const SIMD_WIDTH: usize = 16;
    let len = input.len();
    let mut i = 0;
    
    // Process 16 elements at a time
    while i + SIMD_WIDTH <= len {
        let x = unsafe { _mm512_loadu_ps(input.as_ptr().add(i)) };
        
        // Use the helper tanh approximation
        let result = unsafe { tanh_approx_avx512(x) };
        
        unsafe { _mm512_storeu_ps(output.as_mut_ptr().add(i), result) };
        i += SIMD_WIDTH;
    }
    
    // Handle remaining elements
    for j in i..len {
        output[j] = input[j].tanh();
    }
}

/// AVX-512 GELU activation
#[target_feature(enable = "avx512f")]
pub unsafe fn gelu_f32_avx512(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    const SIMD_WIDTH: usize = 16;
    let len = input.len();
    let mut i = 0;
    
    let sqrt_2_over_pi = _mm512_set1_ps(0.7978845608);
    let coeff = _mm512_set1_ps(0.044715);
    let half = _mm512_set1_ps(0.5);
    let one = _mm512_set1_ps(1.0);
    
    // Process 16 elements at a time
    while i + SIMD_WIDTH <= len {
        let x = unsafe { _mm512_loadu_ps(input.as_ptr().add(i)) };
        
        // x³
        let x_squared = _mm512_mul_ps(x, x);
        let x_cubed = _mm512_mul_ps(x_squared, x);
        
        // sqrt(2/π) * (x + 0.044715 * x³)
        let coeff_x_cubed = _mm512_mul_ps(coeff, x_cubed);
        let inner_sum = _mm512_add_ps(x, coeff_x_cubed);
        let scaled = _mm512_mul_ps(sqrt_2_over_pi, inner_sum);
        
        // Use a more accurate tanh approximation
        let tanh_approx = unsafe { tanh_approx_avx512(scaled) };
        
        // 0.5 * x * (1 + tanh(...))
        let one_plus_tanh = _mm512_add_ps(one, tanh_approx);
        let result = _mm512_mul_ps(_mm512_mul_ps(half, x), one_plus_tanh);
        
        unsafe { _mm512_storeu_ps(output.as_mut_ptr().add(i), result) };
        i += SIMD_WIDTH;
    }
    
    // Handle remaining elements
    for j in i..len {
        let x = input[j];
        let x_cubed = x * x * x;
        let inner = 0.7978845608 * (x + 0.044715 * x_cubed);
        output[j] = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// Helper function for tanh approximation using AVX-512
#[target_feature(enable = "avx512f")]
unsafe fn tanh_approx_avx512(x: __m512) -> __m512 {
    // Clamp to avoid overflow
    let clamp_val = _mm512_set1_ps(10.0);
    let neg_clamp_val = _mm512_set1_ps(-10.0);
    let x_clamped = _mm512_max_ps(_mm512_min_ps(x, clamp_val), neg_clamp_val);
    
    // Rational approximation: tanh(x) ≈ x * (27 + x²) / (27 + 9*x²)
    let x_sq = _mm512_mul_ps(x_clamped, x_clamped);
    let twenty_seven = _mm512_set1_ps(27.0);
    let nine = _mm512_set1_ps(9.0);
    
    let numerator = _mm512_add_ps(twenty_seven, x_sq);
    let denominator = _mm512_add_ps(twenty_seven, _mm512_mul_ps(nine, x_sq));
    
    let ratio = _mm512_div_ps(numerator, denominator);
    _mm512_mul_ps(x_clamped, ratio)
}

/// AVX-512 INT8 dot product (simplified version without VNNI for broader compatibility)
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_product_i8_avx512(
    a: &[i8],
    b: &[i8],
    output: &mut [i32],
    batch_size: usize,
    feature_size: usize,
) {
    assert_eq!(a.len(), batch_size * feature_size);
    assert_eq!(b.len(), feature_size);
    assert_eq!(output.len(), batch_size);
    
    for batch in 0..batch_size {
        let mut sum = 0i32;
        let a_offset = batch * feature_size;
        
        // Use scalar computation for now (VNNI requires special handling)
        for i in 0..feature_size {
            sum += (a[a_offset + i] as i32) * (b[i] as i32);
        }
        
        output[batch] = sum;
    }
}

/// Simple hash computation using AVX-512 (demonstration function)
#[target_feature(enable = "avx512f")]
pub unsafe fn hash_computation_avx512(data: &[u32], output: &mut [u32]) {
    assert_eq!(data.len(), output.len());
    
    const SIMD_WIDTH: usize = 16;
    let len = data.len();
    let mut i = 0;
    
    let multiplier = _mm512_set1_epi32(0x9e3779b9u32 as i32); // Golden ratio hash multiplier
    
    // Process 16 elements at a time
    while i + SIMD_WIDTH <= len {
        let vec = unsafe { _mm512_loadu_epi32(data.as_ptr().add(i) as *const i32) };
        let hashed = _mm512_mullo_epi32(vec, multiplier);
        unsafe { _mm512_storeu_epi32(output.as_mut_ptr().add(i) as *mut i32, hashed) };
        i += SIMD_WIDTH;
    }
    
    // Handle remaining elements
    for j in i..len {
        output[j] = data[j].wrapping_mul(0x9e3779b9u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::fallback;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_avx512_relu() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        
        let input: Vec<f32> = (-16..16).map(|x| x as f32 * 0.5).collect();
        let mut avx512_output = vec![0.0; input.len()];
        let mut scalar_output = vec![0.0; input.len()];
        
        unsafe {
            relu_f32_avx512(&input, &mut avx512_output);
        }
        fallback::relu_f32(&input, &mut scalar_output);
        
        for (avx512, scalar) in avx512_output.iter().zip(scalar_output.iter()) {
            assert_relative_eq!(avx512, scalar, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_avx512_dot_product_i8() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        
        let a: Vec<i8> = (0..128).map(|x| (x % 127) as i8).collect();
        let b: Vec<i8> = (0..64).map(|x| (x % 127) as i8).collect();
        let mut output = [0i32; 2];
        
        // Reshape to 2 batches of 64 features each
        unsafe {
            dot_product_i8_avx512(&a, &b, &mut output, 2, 64);
        }
        
        // Verify with scalar computation
        let mut expected = [0i32; 2];
        for batch in 0..2 {
            let mut sum = 0i32;
            for i in 0..64 {
                sum += (a[batch * 64 + i] as i32) * (b[i] as i32);
            }
            expected[batch] = sum;
        }
        
        assert_eq!(output, expected);
    }
}