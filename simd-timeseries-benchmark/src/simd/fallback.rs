//! Scalar fallback implementations
//!
//! This module provides scalar implementations of all operations as fallbacks
//! when SIMD instructions are not available.
//!
//! All functions in this module are `no_std` compatible and provide reference
//! implementations for correctness testing of SIMD variants.

/// Scalar matrix multiplication with bounds checking
/// 
/// Performs C = A * B where:
/// - A is m×k matrix
/// - B is k×n matrix  
/// - C is m×n matrix
/// 
/// # Arguments
/// * `a` - Left matrix in row-major order
/// * `b` - Right matrix in row-major order
/// * `c` - Output matrix in row-major order
/// * `m`, `n`, `k` - Matrix dimensions
/// 
/// # Returns
/// Returns `Ok(())` on success, or `Err` if dimensions are invalid.
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> crate::Result<()> {
    // Validate input dimensions
    if a.len() != m * k {
        return Err(crate::Error::InvalidDimensions {
            message: format!("Matrix A dimension mismatch: expected {}×{} ({}), got {}", 
                           m, k, m * k, a.len())
        });
    }
    if b.len() != k * n {
        return Err(crate::Error::InvalidDimensions {
            message: format!("Matrix B dimension mismatch: expected {}×{} ({}), got {}", 
                           k, n, k * n, b.len())
        });
    }
    if c.len() != m * n {
        return Err(crate::Error::InvalidDimensions {
            message: format!("Matrix C dimension mismatch: expected {}×{} ({}), got {}", 
                           m, n, m * n, c.len())
        });
    }
    
    // Perform matrix multiplication: C = A * B
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    
    Ok(())
}

/// Scalar ReLU activation function
/// 
/// Computes ReLU(x) = max(0, x) element-wise.
/// 
/// # Arguments
/// * `input` - Input array
/// * `output` - Output array (must be same length as input)
pub fn relu_f32(input: &[f32], output: &mut [f32]) -> crate::Result<()> {
    if input.len() != output.len() {
        return Err(crate::Error::InvalidDimensions {
            message: format!("Input and output arrays must have the same length: {} vs {}", 
                           input.len(), output.len())
        });
    }
    
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = inp.max(0.0f32);
    }
    
    Ok(())
}

/// Scalar tanh activation function
/// 
/// Computes tanh(x) element-wise.
/// 
/// # Arguments
/// * `input` - Input array
/// * `output` - Output array (must be same length as input)
pub fn tanh_f32(input: &[f32], output: &mut [f32]) -> crate::Result<()> {
    if input.len() != output.len() {
        return Err(crate::Error::InvalidDimensions {
            message: format!("Input and output arrays must have the same length: {} vs {}", 
                           input.len(), output.len())
        });
    }
    
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = inp.tanh();
    }
    
    Ok(())
}

/// Scalar GELU activation function
/// 
/// Computes GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// 
/// This is the original GELU formulation using tanh approximation.
/// 
/// # Arguments
/// * `input` - Input array
/// * `output` - Output array (must be same length as input)
pub fn gelu_f32(input: &[f32], output: &mut [f32]) -> crate::Result<()> {
    if input.len() != output.len() {
        return Err(crate::Error::InvalidDimensions {
            message: format!("Input and output arrays must have the same length: {} vs {}", 
                           input.len(), output.len())
        });
    }
    
    const SQRT_2_OVER_PI: f32 = 0.797_884_56; // sqrt(2/π)
    const COEFF: f32 = 0.044_715;
    
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let x = *inp;
        let x_cubed = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
        *out = 0.5f32 * x * (1.0f32 + inner.tanh());
    }
    
    Ok(())
}

/// Scalar Swish/SiLU activation function
/// 
/// Computes Swish(x) = x * sigmoid(x) = x / (1 + exp(-x)) element-wise.
/// 
/// # Arguments
/// * `input` - Input array
/// * `output` - Output array (must be same length as input)
pub fn swish_f32(input: &[f32], output: &mut [f32]) -> crate::Result<()> {
    if input.len() != output.len() {
        return Err(crate::Error::InvalidDimensions {
            message: format!("Input and output arrays must have the same length: {} vs {}", 
                           input.len(), output.len())
        });
    }
    
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let x = *inp;
        *out = x / (1.0f32 + (-x).exp());
    }
    
    Ok(())
}

/// Scalar softmax activation function
/// 
/// Computes softmax(x) = exp(x_i) / sum(exp(x_j)) for all j, with numerical stability.
/// 
/// # Arguments
/// * `input` - Input array
/// * `output` - Output array (must be same length as input)
pub fn softmax_f32(input: &[f32], output: &mut [f32]) -> crate::Result<()> {
    if input.len() != output.len() {
        return Err(crate::Error::InvalidDimensions {
            message: format!("Input and output arrays must have the same length: {} vs {}", 
                           input.len(), output.len())
        });
    }
    
    if input.is_empty() {
        return Ok(());
    }
    
    // Find maximum for numerical stability
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    
    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let exp_val = (inp - max_val).exp();
        *out = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    if sum > 0.0f32 {
        for out in output.iter_mut() {
            *out /= sum;
        }
    }
    
    Ok(())
}

/// Scalar 1D convolution with error checking
/// 
/// Performs 1D convolution: output[i] = sum(input[i*stride + j] * kernel[j]) for j in kernel_len.
/// 
/// # Arguments
/// * `input` - Input array
/// * `kernel` - Convolution kernel
/// * `output` - Output array
/// * `input_len` - Length of input (must match input.len())
/// * `kernel_len` - Length of kernel (must match kernel.len())
/// * `stride` - Convolution stride
pub fn conv1d_f32(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    input_len: usize,
    kernel_len: usize,
    stride: usize,
) -> crate::Result<()> {
    // Validate inputs
    if input.len() != input_len {
        return Err(crate::Error::InvalidDimensions {
            message: format!("Input length mismatch: expected {}, got {}", input_len, input.len())
        });
    }
    
    if kernel.len() != kernel_len {
        return Err(crate::Error::InvalidDimensions {
            message: format!("Kernel length mismatch: expected {}, got {}", kernel_len, kernel.len())
        });
    }
    
    if stride == 0 {
        return Err(crate::Error::InvalidDimensions {
            message: "Stride must be greater than 0".to_string()
        });
    }
    
    if kernel_len > input_len {
        return Err(crate::Error::InvalidDimensions {
            message: format!("Kernel length ({}) cannot be greater than input length ({})", 
                           kernel_len, input_len)
        });
    }
    
    let expected_output_len = (input_len - kernel_len) / stride + 1;
    if output.len() != expected_output_len {
        return Err(crate::Error::InvalidDimensions {
            message: format!("Output length mismatch: expected {}, got {}", 
                           expected_output_len, output.len())
        });
    }
    
    // Perform convolution
    for i in 0..expected_output_len {
        let mut sum = 0.0f32;
        let start_idx = i * stride;
        
        for j in 0..kernel_len {
            sum += input[start_idx + j] * kernel[j];
        }
        
        output[i] = sum;
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_relu() {
        let input = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = [0.0; 5];
        relu_f32(&input, &mut output).unwrap();
        assert_eq!(output, [0.0, 0.0, 0.0, 1.0, 2.0]);
    }
    
    #[test]
    fn test_gelu() {
        let input = [0.0, 1.0, -1.0];
        let mut output = [0.0; 3];
        gelu_f32(&input, &mut output).unwrap();
        
        // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        assert_relative_eq!(output[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 0.8413447, epsilon = 1e-6);
        assert_relative_eq!(output[2], -0.15865526, epsilon = 1e-6);
    }
    
    #[test]
    fn test_matmul_simple() {
        // Test 2x2 * 2x2 = 2x2 matrix multiplication
        let a = [1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let b = [5.0, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]
        let mut c = [0.0; 4];
        
        matmul_f32(&a, &b, &mut c, 2, 2, 2).unwrap();
        
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }
    
    #[test]
    fn test_swish() {
        let input = [0.0, 1.0, -1.0, 2.0];
        let mut output = [0.0; 4];
        swish_f32(&input, &mut output).unwrap();
        
        // Swish(0) = 0, Swish(1) ≈ 0.731, Swish(-1) ≈ -0.269, Swish(2) ≈ 1.762
        assert_relative_eq!(output[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 0.7310586, epsilon = 1e-6);
        assert_relative_eq!(output[2], -0.26894143, epsilon = 1e-6);
        assert_relative_eq!(output[3], 1.7615942, epsilon = 1e-6);
    }
    
    #[test] 
    fn test_softmax() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];
        softmax_f32(&input, &mut output).unwrap();
        
        // Should sum to 1.0
        let sum: f32 = output.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        
        // Values should be in descending order (since input is ascending)
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }
    
    #[test]
    fn test_conv1d() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = [0.5, 1.0, 0.5];
        let mut output = [0.0; 3];
        
        conv1d_f32(&input, &kernel, &mut output, 5, 3, 1).unwrap();
        
        // Expected: [0.5*1 + 1.0*2 + 0.5*3, 0.5*2 + 1.0*3 + 0.5*4, 0.5*3 + 1.0*4 + 0.5*5]
        // = [4.0, 5.0, 6.0]
        assert_relative_eq!(output[0], 4.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 5.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 6.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_error_handling() {
        let input = [1.0, 2.0, 3.0];
        let mut wrong_output = [0.0; 2]; // Wrong size
        
        let result = relu_f32(&input, &mut wrong_output);
        assert!(result.is_err());
        
        let result = gelu_f32(&input, &mut wrong_output);
        assert!(result.is_err());
        
        let result = swish_f32(&input, &mut wrong_output);
        assert!(result.is_err());
        
        let result = softmax_f32(&input, &mut wrong_output);
        assert!(result.is_err());
        
        // Test matrix dimension errors
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let mut c = [0.0; 4];
        
        let result = matmul_f32(&a, &b, &mut c, 2, 2, 2); // Wrong dimensions
        assert!(result.is_err());
        
        // Test conv1d errors
        let input = [1.0, 2.0, 3.0];
        let kernel = [0.5, 1.0];
        let mut output = [0.0; 1];
        
        let result = conv1d_f32(&input, &kernel, &mut output, 3, 2, 0); // Zero stride
        assert!(result.is_err());
        
        let mut wrong_output = [0.0; 5];
        let result = conv1d_f32(&input, &kernel, &mut wrong_output, 3, 2, 1); // Wrong output size
        assert!(result.is_err());
    }
}