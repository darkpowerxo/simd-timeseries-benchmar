//! 1D Convolution implementations
//!
//! High-performance 1D convolution operations for time series processing.

use crate::{Result, Error};

/// 1D convolution with automatic backend selection
pub fn conv1d_f32(
    input: &[f32],
    kernel: &[f32], 
    output: &mut [f32],
    input_len: usize,
    kernel_len: usize,
    stride: usize,
) -> Result<()> {
    let expected_output_len = (input_len - kernel_len) / stride + 1;
    
    if input.len() != input_len || output.len() != expected_output_len {
        return Err(Error::InvalidDimensions {
            message: format!("Conv1D dimension mismatch: input len {}, kernel len {}, output len {} (expected {})",
                           input.len(), kernel_len, output.len(), expected_output_len)
        });
    }
    
    // Currently using scalar fallback - SIMD versions will be implemented in later tasks
    crate::simd::fallback::conv1d_f32(input, kernel, output, input_len, kernel_len, stride);
    Ok(())
}

/// Depthwise 1D convolution (placeholder)
pub fn depthwise_conv1d_f32(
    input: &[f32],
    kernels: &[f32],
    output: &mut [f32], 
    channels: usize,
    input_len: usize,
    kernel_len: usize,
    stride: usize,
) -> Result<()> {
    // Placeholder - will implement proper depthwise convolution later
    let output_len = (input_len - kernel_len) / stride + 1;
    
    if input.len() != channels * input_len || 
       kernels.len() != channels * kernel_len ||
       output.len() != channels * output_len {
        return Err(Error::InvalidDimensions {
            message: "Depthwise conv1D dimension mismatch".to_string()
        });
    }
    
    // Simple channel-wise processing
    for c in 0..channels {
        let input_slice = &input[c * input_len..(c + 1) * input_len];
        let kernel_slice = &kernels[c * kernel_len..(c + 1) * kernel_len];
        let output_slice = &mut output[c * output_len..(c + 1) * output_len];
        
        conv1d_f32(input_slice, kernel_slice, output_slice, input_len, kernel_len, stride)?;
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_conv1d_basic() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = [1.0, -1.0]; // Simple edge detection
        let mut output = [0.0; 4];
        
        conv1d_f32(&input, &kernel, &mut output, 5, 2, 1).unwrap();
        
        // Expected: [1*1 + 2*(-1), 2*1 + 3*(-1), 3*1 + 4*(-1), 4*1 + 5*(-1)]
        //         = [-1, -1, -1, -1]
        let expected = [-1.0, -1.0, -1.0, -1.0];
        
        for (actual, exp) in output.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, exp, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_conv1d_stride() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let kernel = [1.0, 1.0];
        let mut output = [0.0; 2];
        
        conv1d_f32(&input, &kernel, &mut output, 6, 2, 2).unwrap();
        
        // With stride 2: positions 0 and 2
        // [1*1 + 2*1, 3*1 + 4*1] = [3, 7]
        let expected = [3.0, 7.0];
        
        for (actual, exp) in output.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, exp, epsilon = 1e-6);
        }
    }
}