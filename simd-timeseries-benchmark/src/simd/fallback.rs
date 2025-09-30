//! Scalar fallback implementations
//!
//! This module provides scalar implementations of all operations as fallbacks
//! when SIMD instructions are not available.

/// Scalar matrix multiplication
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Scalar ReLU activation
pub fn relu_f32(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = inp.max(0.0);
    }
}

/// Scalar tanh activation
pub fn tanh_f32(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = inp.tanh();
    }
}

/// Scalar GELU activation
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub fn gelu_f32(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/π)
    const COEFF: f32 = 0.044715;
    
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let x = *inp;
        let x_cubed = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
        *out = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// Scalar Swish/SiLU activation
/// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
pub fn swish_f32(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let x = *inp;
        *out = x / (1.0 + (-x).exp());
    }
}

/// Scalar softmax
pub fn softmax_f32(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    // Find maximum for numerical stability
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    
    // Compute exp(x - max) and sum
    let mut sum = 0.0;
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let exp_val = (inp - max_val).exp();
        *out = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for out in output.iter_mut() {
        *out /= sum;
    }
}

/// Scalar 1D convolution
pub fn conv1d_f32(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    input_len: usize,
    kernel_len: usize,
    stride: usize,
) {
    let output_len = (input_len - kernel_len) / stride + 1;
    assert_eq!(output.len(), output_len);
    
    for i in 0..output_len {
        let mut sum = 0.0;
        let start_idx = i * stride;
        
        for j in 0..kernel_len {
            sum += input[start_idx + j] * kernel[j];
        }
        
        output[i] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_relu() {
        let input = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = [0.0; 5];
        relu_f32(&input, &mut output);
        assert_eq!(output, [0.0, 0.0, 0.0, 1.0, 2.0]);
    }
    
    #[test]
    fn test_gelu() {
        let input = [0.0, 1.0, -1.0];
        let mut output = [0.0; 3];
        gelu_f32(&input, &mut output);
        
        // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        assert_relative_eq!(output[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 0.8413447, epsilon = 1e-6);
        assert_relative_eq!(output[2], -0.15865526, epsilon = 1e-6);
    }
}