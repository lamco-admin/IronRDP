//! ZGFX Compressor
//!
//! Full implementation of ZGFX (RDP8) compression algorithm.
//!
//! # Algorithm Overview
//!
//! ZGFX uses an LZ77-variant algorithm with Huffman-style token encoding:
//! - Maintains a 2.5MB circular history buffer
//! - Finds matching sequences in the history using hash table (O(1) lookups)
//! - Encodes data as literals or back-references (matches)
//! - Uses variable-length prefix codes for efficient encoding
//!
//! # Performance Optimization
//!
//! This implementation uses a hash table to accelerate match finding:
//! - Maps 3-byte prefixes to positions in history buffer
//! - O(1) lookup instead of O(n) linear scan
//! - Dramatically improves compression speed (100-1000x faster)
//!
//! # Token Types
//!
//! 1. **Null Literal**: Prefix "0" + 8 bits → any byte value
//! 2. **Literal Tokens**: Short prefixes for common bytes (0x00, 0x01, 0xFF, etc.)
//! 3. **Match Tokens**: Back-reference with distance and length encoding
//!
//! # Match Encoding
//!
//! - Distance: Encoded using match token + additional value bits
//! - Length: Variable-length encoding (special case for length=3)
//!
//! # References
//!
//! - MS-RDPEGFX Section 2.2.1.1.1: ZGFX Compression Algorithm

use std::collections::HashMap;

use bitvec::prelude::*;

use super::ZgfxError;
use super::TOKEN_TABLE;

const HISTORY_SIZE: usize = 2_500_000;
const MIN_MATCH_LENGTH: usize = 3;
const MAX_MATCH_LENGTH: usize = 65535; // Practical limit
const MAX_MATCH_DISTANCE: usize = 2_097_152; // Max for last token

/// Limits worst-case performance when many positions share the same 3-byte prefix
const MAX_CANDIDATES: usize = 16;

/// Prevents unbounded growth for frequently occurring prefixes
const MAX_POSITIONS_PER_PREFIX: usize = 32;

/// Triggers cleanup to keep memory usage bounded
const MAX_HASH_TABLE_ENTRIES: usize = 50_000;

/// ZGFX Compressor with history buffer and hash table for fast match finding
pub struct Compressor {
    /// Previously compressed data that can be referenced by back-references
    history: Vec<u8>,

    /// Maps 3-byte prefixes to history positions for O(1) match lookup
    /// (avoids O(n) scan of 2.5MB history buffer)
    match_table: HashMap<[u8; 3], Vec<usize>>,
}

impl Compressor {
    pub fn new() -> Self {
        Self {
            history: Vec::with_capacity(HISTORY_SIZE),
            match_table: HashMap::new(),
        }
    }

    /// Compress data using ZGFX algorithm.
    ///
    /// Returns compressed data ready to be wrapped in ZGFX segment structure.
    pub fn compress(&mut self, input: &[u8]) -> Result<Vec<u8>, ZgfxError> {
        let mut bit_writer = BitWriter::new();
        let mut pos = 0;

        while pos < input.len() {
            // Matches compress better than literals - worth the lookup cost
            let best_match = self.find_best_match(input, pos);

            if let Some(m) = best_match {
                if m.length >= MIN_MATCH_LENGTH {
                    self.encode_match(&mut bit_writer, m.distance, m.length)?;
                    // Extend history so future input can reference these bytes
                    self.add_to_history(&input[pos..pos + m.length]);
                    pos += m.length;
                    continue;
                }
            }

            let byte = input[pos];
            self.encode_literal(&mut bit_writer, byte)?;
            self.add_to_history(&[byte]);
            pos += 1;
        }

        Ok(bit_writer.finish())
    }

    /// Add bytes to history buffer, managing the 2.5MB sliding window
    fn add_to_history(&mut self, bytes: &[u8]) {
        // ZGFX uses a 2.5MB sliding window - drop oldest bytes when full
        if self.history.len() + bytes.len() > HISTORY_SIZE {
            let overflow = (self.history.len() + bytes.len()) - HISTORY_SIZE;

            self.history.drain(..overflow);

            // Shift hash table positions to account for removed bytes
            for positions in self.match_table.values_mut() {
                positions.retain_mut(|pos| {
                    if *pos >= overflow {
                        *pos -= overflow;
                        true
                    } else {
                        false
                    }
                });
            }

            // Prevent hash table from accumulating stale empty vectors
            self.match_table.retain(|_, positions| !positions.is_empty());
        }

        // Must update history before hash table to ensure consistent positions
        let base_pos = self.history.len();
        self.history.extend_from_slice(bytes);

        // CRITICAL: Only add each position ONCE to avoid duplicates that cause
        // exponential slowdown (bug fix: positions were being added multiple times)

        // OPTIMIZATION: For large chunks (matches), sample every 4th position
        // to keep hash table manageable while maintaining good compression
        let step_size = if bytes.len() > 256 { 4 } else { 1 };

        for i in (0..bytes.len().saturating_sub(MIN_MATCH_LENGTH - 1)).step_by(step_size) {
            let pos = base_pos + i;
            let prefix = [
                self.history[pos],
                self.history[pos + 1],
                self.history[pos + 2],
            ];

            let entry = self.match_table
                .entry(prefix)
                .or_insert_with(Vec::new);

            if entry.len() < MAX_POSITIONS_PER_PREFIX {
                entry.push(pos);
            } else {
                // Sliding window: keep most recent positions for better compression
                entry.remove(0);
                entry.push(pos);
            }
        }

        if self.match_table.len() > MAX_HASH_TABLE_ENTRIES {
            self.compact_hash_table();
        }

        // Handle sequences that span the old/new boundary
        if base_pos >= 2 && bytes.len() >= 1 {
            let pos = base_pos - 2;
            if pos + MIN_MATCH_LENGTH <= self.history.len() {
                let prefix = [self.history[pos], self.history[pos + 1], self.history[pos + 2]];
                let entry = self.match_table.entry(prefix).or_insert_with(Vec::new);
                if entry.last() != Some(&pos) {
                    entry.push(pos);
                }
            }
        }
        if base_pos >= 1 && bytes.len() >= 2 {
            let pos = base_pos - 1;
            if pos + MIN_MATCH_LENGTH <= self.history.len() {
                let prefix = [self.history[pos], self.history[pos + 1], self.history[pos + 2]];
                let entry = self.match_table.entry(prefix).or_insert_with(Vec::new);
                if entry.last() != Some(&pos) {
                    entry.push(pos);
                }
            }
        }
    }

    /// Keep only recent positions to bound memory usage
    fn compact_hash_table(&mut self) {
        for positions in self.match_table.values_mut() {
            if positions.len() > MAX_POSITIONS_PER_PREFIX / 2 {
                let keep_from = positions.len() - (MAX_POSITIONS_PER_PREFIX / 2);
                *positions = positions[keep_from..].to_vec();
            }
        }

        self.match_table.retain(|_, positions| !positions.is_empty());
    }

    /// Find best match using hash table for O(1) candidate lookup
    ///
    /// This is performance-critical: hash table avoids O(n) scan of 2.5MB history.
    fn find_best_match(&self, input: &[u8], pos: usize) -> Option<Match> {
        let remaining = input.len() - pos;
        if remaining < MIN_MATCH_LENGTH || self.history.is_empty() {
            return None;
        }

        // 3 bytes = minimum match length per ZGFX spec
        let prefix = [input[pos], input[pos + 1], input[pos + 2]];

        let candidates = self.match_table.get(&prefix)?;

        let max_match_len = remaining.min(MAX_MATCH_LENGTH);
        let mut best_match: Option<Match> = None;
        let search_limit = self.history.len().min(MAX_MATCH_DISTANCE);

        // Check most recent candidates first (better locality, often better matches)
        for &hist_pos in candidates.iter().rev().take(MAX_CANDIDATES) {
            let distance = self.history.len() - hist_pos;

            if distance > search_limit {
                continue;
            }

            // First 3 bytes already match (that's how we found this candidate)
            let mut match_len = MIN_MATCH_LENGTH;

            while match_len < max_match_len
                && hist_pos + match_len < self.history.len()
                && self.history[hist_pos + match_len] == input[pos + match_len]
            {
                match_len += 1;
            }

            if let Some(ref current_best) = best_match {
                if match_len > current_best.length {
                    best_match = Some(Match { distance, length: match_len });
                }
            } else {
                best_match = Some(Match { distance, length: match_len });
            }

            // Diminishing returns beyond 32 bytes - stop early
            if match_len >= 32 {
                break;
            }
        }

        best_match
    }

    /// Select appropriate token from MS-RDPEGFX encoding table
    fn find_match_token(distance: usize) -> MatchToken {
        // Tokens 26-39 encode matches with increasing distance ranges
        for token in TOKEN_TABLE.iter().skip(26) {
            if let super::TokenType::Match {
                distance_value_size,
                distance_base,
            } = token.ty
            {
                let max_distance = distance_base as usize + (1 << distance_value_size) - 1;
                if distance <= max_distance {
                    return MatchToken {
                        prefix: token.prefix,
                        distance_value_size,
                        distance_base: distance_base as usize,
                    };
                }
            }
        }

        // Last token covers distances up to 2MB (full history range)
        if let super::TokenType::Match {
            distance_value_size,
            distance_base,
        } = TOKEN_TABLE[39].ty
        {
            MatchToken {
                prefix: TOKEN_TABLE[39].prefix,
                distance_value_size,
                distance_base: distance_base as usize,
            }
        } else {
            unreachable!("Last token must be Match type");
        }
    }

    /// Check if byte has a dedicated short encoding
    fn find_literal_token(byte: u8) -> Option<usize> {
        // Tokens 1-25 provide shorter encodings for common byte values
        for (i, token) in TOKEN_TABLE.iter().enumerate().take(26).skip(1) {
            if let super::TokenType::Literal { literal_value } = token.ty {
                if literal_value == byte {
                    return Some(i);
                }
            }
        }
        None
    }

    fn encode_literal(&self, writer: &mut BitWriter, byte: u8) -> Result<(), ZgfxError> {
        // Common bytes (0x00, 0xFF, etc.) have shorter dedicated tokens
        if let Some(token_idx) = Self::find_literal_token(byte) {
            let token = &TOKEN_TABLE[token_idx];
            writer.write_bits_from_slice(token.prefix);
        } else {
            // Null literal: "0" prefix + 8-bit value (fallback for uncommon bytes)
            writer.write_bit(false);
            writer.write_bits(byte as u32, 8);
        }
        Ok(())
    }

    fn encode_match(&self, writer: &mut BitWriter, distance: usize, length: usize) -> Result<(), ZgfxError> {
        let match_token = Self::find_match_token(distance);

        writer.write_bits_from_slice(match_token.prefix);

        let distance_value = distance - match_token.distance_base;
        writer.write_bits(distance_value as u32, match_token.distance_value_size);

        self.encode_match_length(writer, length)?;

        Ok(())
    }

    fn encode_match_length(&self, writer: &mut BitWriter, length: usize) -> Result<(), ZgfxError> {
        if length == 3 {
            // Length 3 has special single-bit encoding per spec
            writer.write_bit(false);
        } else {
            // Variable-length encoding: length = 2^(token_size+1) + value
            let length_token_size = (length as f64).log2().floor() as usize - 1;
            let base = 1 << (length_token_size + 1);
            let value = length - base;

            for _ in 0..length_token_size {
                writer.write_bit(true);
            }
            writer.write_bit(false);

            writer.write_bits(value as u32, length_token_size + 1);
        }

        Ok(())
    }
}

impl Default for Compressor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
struct Match {
    distance: usize,
    length: usize,
}

struct MatchToken {
    prefix: &'static BitSlice<u8, Msb0>,
    distance_value_size: usize,
    distance_base: usize,
}

/// Bit-level writer for ZGFX token encoding (MSB first)
struct BitWriter {
    bytes: Vec<u8>,
    current_byte: u8,
    bits_in_current: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current_byte: 0,
            bits_in_current: 0,
        }
    }

    fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.bits_in_current);
        }
        self.bits_in_current += 1;

        if self.bits_in_current == 8 {
            self.bytes.push(self.current_byte);
            self.current_byte = 0;
            self.bits_in_current = 0;
        }
    }

    fn write_bits(&mut self, value: u32, num_bits: usize) {
        for i in (0..num_bits).rev() {
            let bit = (value >> i) & 1 == 1;
            self.write_bit(bit);
        }
    }

    fn write_bits_from_slice(&mut self, bits: &BitSlice<u8, Msb0>) {
        for bit in bits {
            self.write_bit(*bit);
        }
    }

    /// Finalize output with ZGFX-required padding indicator byte
    fn finish(mut self) -> Vec<u8> {
        let unused_bits = if self.bits_in_current == 0 {
            0
        } else {
            8 - self.bits_in_current
        };

        if self.bits_in_current > 0 {
            self.bytes.push(self.current_byte);
        }

        // ZGFX format requires trailing byte indicating unused bits in final byte
        self.bytes.push(unused_bits as u8);

        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_new() {
        let compressor = Compressor::new();
        assert_eq!(compressor.history.len(), 0);
    }

    #[test]
    fn test_compress_empty() {
        let mut compressor = Compressor::new();
        let compressed = compressor.compress(&[]).unwrap();

        assert_eq!(compressed.len(), 1);
        assert_eq!(compressed[0], 0);
    }

    #[test]
    fn test_compress_single_byte() {
        let mut compressor = Compressor::new();
        let compressed = compressor.compress(&[0x42]).unwrap();

        // Null literal: "0" + 8 bits + padding = 9 bits = 2 bytes + padding byte
        assert!(compressed.len() >= 2);
    }

    #[test]
    fn test_compress_round_trip() {
        use super::super::Decompressor;

        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        let data = b"Hello, ZGFX compression! This is a test.";
        let compressed = compressor.compress(data).unwrap();

        let mut output = Vec::new();
        decompressor.decompress_segment(&compressed, &mut output).unwrap();

        assert_eq!(&output, data);
    }

    #[test]
    fn test_compress_with_repetition() {
        use super::super::Decompressor;

        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        let data = b"AAAAAAAAAABBBBBBBBBBCCCCCCCCCC";
        let compressed = compressor.compress(data).unwrap();

        let mut output = Vec::new();
        decompressor.decompress_segment(&compressed, &mut output).unwrap();

        assert_eq!(&output, data);

        println!(
            "Original: {} bytes, Compressed: {} bytes, Ratio: {:.2}x",
            data.len(),
            compressed.len(),
            data.len() as f64 / compressed.len() as f64
        );
    }

    #[test]
    fn test_compress_large_data() {
        use super::super::Decompressor;

        let mut compressor = Compressor::new();
        let mut decompressor = Decompressor::new();

        let mut data = Vec::new();
        for i in 0..1000 {
            data.extend_from_slice(b"Pattern");
            data.push((i % 256) as u8);
        }

        let compressed = compressor.compress(&data).unwrap();

        let mut output = Vec::new();
        decompressor.decompress_segment(&compressed, &mut output).unwrap();

        assert_eq!(output, data);

        println!(
            "Original: {} bytes, Compressed: {} bytes, Ratio: {:.2}x",
            data.len(),
            compressed.len(),
            data.len() as f64 / compressed.len() as f64
        );
    }

    #[test]
    fn test_bit_writer() {
        let mut writer = BitWriter::new();

        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bits(0b101, 3);

        let result = writer.finish();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0b10110100);
        assert_eq!(result[1], 2);
    }

    #[test]
    fn test_encode_literal_with_token() {
        let compressor = Compressor::new();
        let mut writer = BitWriter::new();

        compressor.encode_literal(&mut writer, 0x00).unwrap();

        let result = writer.finish();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_encode_literal_null() {
        let compressor = Compressor::new();
        let mut writer = BitWriter::new();

        compressor.encode_literal(&mut writer, 0x42).unwrap();

        let result = writer.finish();
        assert_eq!(result.len(), 3);
        assert_eq!(result[2], 7);
    }
}
