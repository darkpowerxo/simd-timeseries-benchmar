//! MLP with sliding window for time series forecasting
//!
//! Low-latency MLP implementation with windowing approach.

use crate::{Result, ops};

/// Simple MLP layer
pub struct MLPLayer {
    weights: Vec<f32>,
    biases: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

impl MLPLayer {
    /// Create a new MLP layer
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Initialize with small random weights (placeholder)
        let weights = vec![0.1; input_size * output_size];
        let biases = vec![0.0; output_size];
        
        MLPLayer {
            weights,
            biases,
            input_size,
            output_size,
        }
    }
    
    /// Forward pass through the layer
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        // Simple matrix multiplication followed by bias addition
        ops::matmul_f32(
            input, 
            &self.weights, 
            output,
            1, // batch size 1
            self.output_size,
            self.input_size,
        )?;
        
        // Add biases
        for (out, bias) in output.iter_mut().zip(self.biases.iter()) {
            *out += bias;
        }
        
        Ok(())
    }
}

/// MLP with windowing for time series
pub struct WindowedMLP {
    layers: Vec<MLPLayer>,
    window_size: usize,
}

impl WindowedMLP {
    /// Create a new windowed MLP
    pub fn new(window_size: usize, hidden_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        
        if !hidden_sizes.is_empty() {
            // Input layer
            layers.push(MLPLayer::new(window_size, hidden_sizes[0]));
            
            // Hidden layers
            for i in 1..hidden_sizes.len() {
                layers.push(MLPLayer::new(hidden_sizes[i-1], hidden_sizes[i]));
            }
            
            // Output layer (single value prediction)
            layers.push(MLPLayer::new(hidden_sizes[hidden_sizes.len()-1], 1));
        }
        
        WindowedMLP {
            layers,
            window_size,
        }
    }
    
    /// Predict next value given a window of input data
    pub fn predict(&self, window: &[f32]) -> Result<f32> {
        if window.len() != self.window_size {
            return Err(crate::Error::InvalidDimensions {
                message: format!("Window size mismatch: expected {}, got {}", 
                               self.window_size, window.len())
            });
        }
        
        if self.layers.is_empty() {
            return Ok(0.0);
        }
        
        let mut current = window.to_vec();
        let mut next = vec![0.0; self.layers[0].output_size];
        
        // Forward pass through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(&current, &mut next)?;
            
            // Apply ReLU activation (except for output layer)
            if i < self.layers.len() - 1 {
                let mut temp = next.clone();
                ops::relu(&next, &mut temp)?;
                next = temp;
            }
            
            // Prepare for next layer
            if i < self.layers.len() - 1 {
                current = next.clone();
                next = vec![0.0; self.layers[i + 1].output_size];
            }
        }
        
        Ok(next[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    
    #[test]
    fn test_mlp_layer_creation() {
        let layer = MLPLayer::new(4, 8);
        assert_eq!(layer.input_size, 4);
        assert_eq!(layer.output_size, 8);
        assert_eq!(layer.weights.len(), 32);
        assert_eq!(layer.biases.len(), 8);
    }
    
    #[test]
    fn test_windowed_mlp() {
        let mlp = WindowedMLP::new(4, &[8, 4]);
        let window = [1.0, 2.0, 3.0, 4.0];
        
        // Should not panic and return some prediction
        let prediction = mlp.predict(&window).unwrap();
        assert!(prediction.is_finite());
    }
}