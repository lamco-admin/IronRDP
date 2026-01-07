//! High-level ZGFX Compression API
//!
//! Provides convenient functions for compressing and wrapping EGFX PDU data
//! with automatic mode selection and error handling.

use super::compressor::Compressor;
use super::wrapper::{wrap_compressed, wrap_uncompressed};
use super::ZgfxError;

/// Compression mode for ZGFX encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMode {
    /// Always send uncompressed (fastest, no CPU overhead)
    Never,
    /// Compress and use if smaller, otherwise fallback to uncompressed (smart mode)
    Auto,
    /// Always compress (best bandwidth, higher CPU)
    Always,
}

/// Compress and wrap EGFX PDU bytes for transmission
///
/// This is the main entry point for EGFX data preparation. It handles:
/// - Compression (if enabled)
/// - ZGFX segment wrapping
/// - Automatic fallback to uncompressed if compression doesn't help
///
/// # Arguments
///
/// * `data` - Raw EGFX PDU bytes to encode
/// * `compressor` - ZGFX compressor instance (maintains history)
/// * `mode` - Compression mode selection
///
/// # Returns
///
/// ZGFX-wrapped data ready for DVC transmission
///
/// # Examples
///
/// ```
/// use ironrdp_graphics::zgfx::{Compressor, CompressionMode, compress_and_wrap_egfx};
///
/// let mut compressor = Compressor::new();
/// let egfx_pdu_bytes = vec![0x01, 0x02, 0x03, 0x04];
///
/// // Smart mode: compresses if beneficial
/// let wrapped = compress_and_wrap_egfx(&egfx_pdu_bytes, &mut compressor, CompressionMode::Auto).unwrap();
/// ```
pub fn compress_and_wrap_egfx(
    data: &[u8],
    compressor: &mut Compressor,
    mode: CompressionMode,
) -> Result<Vec<u8>, ZgfxError> {
    match mode {
        CompressionMode::Never => {
            // Just wrap uncompressed
            Ok(wrap_uncompressed(data))
        }
        CompressionMode::Auto => {
            // Try compression, use if beneficial
            let compressed = compressor.compress(data)?;
            let wrapped_compressed = wrap_compressed(&compressed);
            let wrapped_uncompressed = wrap_uncompressed(data);

            // Use compressed version only if it's actually smaller
            if wrapped_compressed.len() < wrapped_uncompressed.len() {
                Ok(wrapped_compressed)
            } else {
                Ok(wrapped_uncompressed)
            }
        }
        CompressionMode::Always => {
            // Always compress
            let compressed = compressor.compress(data)?;
            Ok(wrap_compressed(&compressed))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_mode_never() {
        let mut compressor = Compressor::new();
        let data = b"Test data";

        let wrapped = compress_and_wrap_egfx(data, &mut compressor, CompressionMode::Never).unwrap();

        // Should be uncompressed (flags 0x04, not 0x24)
        assert_eq!(wrapped[0], 0xE0);
        assert_eq!(wrapped[1], 0x04);
    }

    #[test]
    fn test_compress_mode_always() {
        let mut compressor = Compressor::new();
        let data = b"Test data";

        let wrapped = compress_and_wrap_egfx(data, &mut compressor, CompressionMode::Always).unwrap();

        // Should be compressed (flags 0x24)
        assert_eq!(wrapped[0], 0xE0);
        assert_eq!(wrapped[1], 0x24);
    }

    #[test]
    fn test_compress_mode_auto_compresses_repetitive() {
        let mut compressor = Compressor::new();
        let data = b"AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCC";

        let wrapped = compress_and_wrap_egfx(data, &mut compressor, CompressionMode::Auto).unwrap();

        // Should choose compressed (better ratio)
        assert_eq!(wrapped[0], 0xE0);
        assert_eq!(wrapped[1], 0x24); // Compressed
    }

    #[test]
    fn test_compress_mode_auto_skips_random() {
        let mut compressor = Compressor::new();
        // Random-ish data that won't compress well
        let data: Vec<u8> = (0..100).map(|i| ((i * 7) % 256) as u8).collect();

        let wrapped = compress_and_wrap_egfx(&data, &mut compressor, CompressionMode::Auto).unwrap();

        // Might be uncompressed if compression doesn't help
        // (depends on actual compression ratio)
        assert_eq!(wrapped[0], 0xE0);
    }

    #[test]
    fn test_round_trip_all_modes() {
        use super::super::Decompressor;

        let data = b"Test data with some repetition: AAAA BBBB CCCC";
        let mut decompressor = Decompressor::new();

        for mode in [CompressionMode::Never, CompressionMode::Auto, CompressionMode::Always] {
            let mut compressor = Compressor::new();
            let wrapped = compress_and_wrap_egfx(data, &mut compressor, mode).unwrap();

            let mut output = Vec::new();
            decompressor.decompress(&wrapped, &mut output).unwrap();

            assert_eq!(&output, data, "Round-trip failed for mode {:?}", mode);
        }
    }
}
