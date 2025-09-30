//! SIMD implementations and feature detection
//! 
//! This module provides runtime CPU feature detection and safe SIMD implementations
//! with automatic fallback to scalar code when SIMD instructions are not available.

use cfg_if::cfg_if;

pub mod fallback;

cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        pub mod avx2;
        pub mod avx512;
        
        /// CPU feature detection results
        #[derive(Debug, Clone, Copy)]
        pub struct Features {
            /// AVX2 support
            pub avx2: bool,
            /// AVX-512 Foundation (F) support
            pub avx512f: bool,
            /// AVX-512 Vector Length (VL) support
            pub avx512vl: bool,
            /// AVX-512 Byte and Word (BW) support
            pub avx512bw: bool,
            /// AVX-512 Doubleword and Quadword (DQ) support
            pub avx512dq: bool,
            /// AVX-512 Conflict Detection (CD) support
            pub avx512cd: bool,
            /// AVX-512 Exponential and Reciprocal (ER) support
            pub avx512er: bool,
            /// AVX-512 Vector Neural Network Instructions (VNNI) support
            pub avx512vnni: bool,
            /// AVX-512 BFloat16 support
            pub avx512bf16: bool,
        }
        
        /// Detect available CPU features at runtime
        pub fn detect_features() -> Features {
            Features {
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512vl: is_x86_feature_detected!("avx512vl"),
                avx512bw: is_x86_feature_detected!("avx512bw"),
                avx512dq: is_x86_feature_detected!("avx512dq"),
                avx512cd: is_x86_feature_detected!("avx512cd"),
                avx512er: is_x86_feature_detected!("avx512er"),
                avx512vnni: is_x86_feature_detected!("avx512vnni"),
                avx512bf16: is_x86_feature_detected!("avx512bf16"),
            }
        }
    } else {
        /// CPU feature detection results (non-x86 platforms)
        #[derive(Debug, Clone, Copy)]
        pub struct Features {
            /// AVX2 support (always false on non-x86)
            pub avx2: bool,
            /// AVX-512 Foundation (F) support (always false on non-x86)
            pub avx512f: bool,
            /// AVX-512 Vector Length (VL) support (always false on non-x86)
            pub avx512vl: bool,
            /// AVX-512 Byte and Word (BW) support (always false on non-x86)
            pub avx512bw: bool,
            /// AVX-512 Doubleword and Quadword (DQ) support (always false on non-x86)
            pub avx512dq: bool,
            /// AVX-512 Conflict Detection (CD) support (always false on non-x86)
            pub avx512cd: bool,
            /// AVX-512 Exponential and Reciprocal (ER) support (always false on non-x86)
            pub avx512er: bool,
            /// AVX-512 Vector Neural Network Instructions (VNNI) support (always false on non-x86)
            pub avx512vnni: bool,
            /// AVX-512 BFloat16 support (always false on non-x86)
            pub avx512bf16: bool,
        }
        
        /// Detect available CPU features at runtime (non-x86 platforms)
        pub fn detect_features() -> Features {
            Features {
                avx2: false,
                avx512f: false,
                avx512vl: false,
                avx512bw: false,
                avx512dq: false,
                avx512cd: false,
                avx512er: false,
                avx512vnni: false,
                avx512bf16: false,
            }
        }
    }
}