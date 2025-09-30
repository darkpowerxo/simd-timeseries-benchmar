//! Video encoding benchmark (AVX-512 F+VL+BW)
//!
//! Byte/word operations for pixel processing.

/// Sum of Absolute Differences (SAD) for motion estimation
pub fn sad_16x16(block1: &[u8], block2: &[u8]) -> u32 {
    assert_eq!(block1.len(), 256); // 16x16 block
    assert_eq!(block2.len(), 256);
    
    let mut sad = 0u32;
    for (p1, p2) in block1.iter().zip(block2.iter()) {
        sad += (*p1 as i16 - *p2 as i16).abs() as u32;
    }
    sad
}

/// Simple DCT (Discrete Cosine Transform) approximation
pub fn dct_8x8(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), 64);  // 8x8 block
    assert_eq!(output.len(), 64);
    
    // Placeholder DCT - just copy for now
    output.copy_from_slice(input);
}

/// Color space conversion RGB to YUV (simplified)
pub fn rgb_to_yuv(rgb: &[u8], yuv: &mut [u8]) {
    assert_eq!(rgb.len(), yuv.len());
    assert_eq!(rgb.len() % 3, 0); // Must be multiple of 3 for RGB pixels
    
    for i in (0..rgb.len()).step_by(3) {
        let r = rgb[i] as f32;
        let g = rgb[i + 1] as f32;
        let b = rgb[i + 2] as f32;
        
        // ITU-R BT.709 conversion
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let u = -0.1146 * r - 0.3854 * g + 0.5 * b + 128.0;
        let v = 0.5 * r - 0.4542 * g - 0.0458 * b + 128.0;
        
        yuv[i] = y.clamp(0.0, 255.0) as u8;
        yuv[i + 1] = u.clamp(0.0, 255.0) as u8;
        yuv[i + 2] = v.clamp(0.0, 255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sad_identical_blocks() {
        let block = vec![128u8; 256];
        let sad = sad_16x16(&block, &block);
        assert_eq!(sad, 0); // Identical blocks should have SAD = 0
    }
    
    #[test]
    fn test_rgb_to_yuv() {
        let rgb = [255, 0, 0, 0, 255, 0, 0, 0, 255]; // Red, Green, Blue pixels
        let mut yuv = [0u8; 9];
        
        rgb_to_yuv(&rgb, &mut yuv);
        
        // Check that conversion produces reasonable values
        assert!(yuv[0] > 0);   // Red should have some Y component
        assert!(yuv[3] > 0);   // Green should have high Y component
        assert!(yuv[6] > 0);   // Blue should have some Y component
    }
}