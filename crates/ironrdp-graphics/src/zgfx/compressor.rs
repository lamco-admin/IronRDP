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

/// Maximum number of candidate positions to check per prefix
/// Limits worst-case performance when many positions share the same prefix
const MAX_CANDIDATES: usize = 16;

/// Maximum positions per hash table entry
/// Prevents unbounded growth for common prefixes
const MAX_POSITIONS_PER_PREFIX: usize = 32;

/// Maximum total hash table entries before cleanup
/// Keeps memory usage bounded
const MAX_HASH_TABLE_ENTRIES: usize = 50_000;

/// ZGFX Compressor with history buffer and hash table for fast match finding
pub struct Compressor {
    /// History buffer containing previously compressed data
    history: Vec<u8>,

    /// Hash table mapping 3-byte prefixes to positions in history
    /// Key: [u8; 3] representing a 3-byte sequence
    /// Value: Vec<usize> of positions where this prefix occurs in history
    match_table: HashMap<[u8; 3], Vec<usize>>,
}

impl Compressor {
    /// Create a new ZGFX compressor
    pub fn new() -> Self {
        Self {
            history: Vec::with_capacity(HISTORY_SIZE),
            match_table: HashMap::new(),
        }
    }

    /// Compress data using ZGFX algorithm
    ///
    /// Returns compressed data with ZGFX token encoding.
    ///
    /// # Arguments
    ///
    /// * `input` - Data to compress
    ///
    /// # Returns
    ///
    /// Compressed data ready to be wrapped in ZGFX segment structure
    pub fn compress(&mut self, input: &[u8]) -> Result<Vec<u8>, ZgfxError> {
        let mut bit_writer = BitWriter::new();
        let mut pos = 0;

        while pos < input.len() {
            // Try to find a match in history
            let best_match = self.find_best_match(input, pos);

            if let Some(m) = best_match {
                if m.length >= MIN_MATCH_LENGTH {
                    // Encode as match
                    self.encode_match(&mut bit_writer, m.distance, m.length)?;

                    // Add matched bytes to history
                    self.add_to_history(&input[pos..pos + m.length]);

                    pos += m.length;
                    continue;
                }
            }

            // Encode as literal
            let byte = input[pos];
            self.encode_literal(&mut bit_writer, byte)?;
            self.add_to_history(&[byte]);
            pos += 1;
        }

        Ok(bit_writer.finish())
    }

    /// Add bytes to history buffer (managing size limit and hash table)
    fn add_to_history(&mut self, bytes: &[u8]) {
        // Handle history buffer overflow
        if self.history.len() + bytes.len() > HISTORY_SIZE {
            let overflow = (self.history.len() + bytes.len()) - HISTORY_SIZE;

            // Remove old bytes from history
            self.history.drain(..overflow);

            // Update hash table: shift all positions down by overflow amount
            // and remove positions that are now negative
            for positions in self.match_table.values_mut() {
                // Update positions, removing those that were drained
                positions.retain_mut(|pos| {
                    if *pos >= overflow {
                        *pos -= overflow;
                        true
                    } else {
                        false // Remove this position
                    }
                });
            }

            // Clean up empty entries to save memory
            self.match_table.retain(|_, positions| !positions.is_empty());
        }

        // Add new bytes to history first
        let base_pos = self.history.len();
        self.history.extend_from_slice(bytes);

        // Update hash table with ONLY truly new 3-byte sequences
        // We add:
        // 1. Sequences that start in the newly added bytes
        // 2. Sequences that span the boundary (start in old, extend into new)
        //
        // CRITICAL: Only add each position ONCE to avoid duplicates

        // Add sequences starting in new bytes
        // OPTIMIZATION: For large chunks (matches), don't add every position
        // Sample positions to keep hash table manageable
        let step_size = if bytes.len() > 256 {
            // For large chunks (matches), sample every 4th position
            4
        } else {
            // For small chunks (literals), add all positions
            1
        };

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

            // Limit positions per prefix to prevent unbounded growth
            if entry.len() < MAX_POSITIONS_PER_PREFIX {
                entry.push(pos);
            } else {
                // Replace oldest position with newest (sliding window)
                entry.remove(0);
                entry.push(pos);
            }
        }

        // Periodic hash table cleanup if it gets too large
        if self.match_table.len() > MAX_HASH_TABLE_ENTRIES {
            self.compact_hash_table();
        }

        // Add sequences that span the boundary
        // Only if we actually added bytes and have prior history
        if base_pos >= 2 && bytes.len() >= 1 {
            let pos = base_pos - 2;
            if pos + MIN_MATCH_LENGTH <= self.history.len() {
                let prefix = [self.history[pos], self.history[pos + 1], self.history[pos + 2]];
                // Check if this position isn't already in the table (avoid duplicates)
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

    /// Compact hash table by keeping only recent positions
    ///
    /// Called when hash table grows too large. Keeps only the most recent
    /// positions for each prefix to maintain performance.
    fn compact_hash_table(&mut self) {
        for positions in self.match_table.values_mut() {
            if positions.len() > MAX_POSITIONS_PER_PREFIX / 2 {
                // Keep only the most recent half
                let keep_from = positions.len() - (MAX_POSITIONS_PER_PREFIX / 2);
                *positions = positions[keep_from..].to_vec();
            }
        }

        // Remove empty entries
        self.match_table.retain(|_, positions| !positions.is_empty());
    }

    /// Find the best match in the history buffer using hash table
    ///
    /// This is the performance-critical function. It uses a hash table to find
    /// candidate positions in O(1) instead of scanning the entire history in O(n).
    ///
    /// Algorithm:
    /// 1. Extract 3-byte prefix from input at current position
    /// 2. Look up candidate positions in hash table (O(1))
    /// 3. Check only those candidates (typically 1-16) for best match
    /// 4. Return longest match found
    fn find_best_match(&self, input: &[u8], pos: usize) -> Option<Match> {
        let remaining = input.len() - pos;
        if remaining < MIN_MATCH_LENGTH || self.history.is_empty() {
            return None;
        }

        // Extract 3-byte prefix for hash table lookup
        let prefix = [input[pos], input[pos + 1], input[pos + 2]];

        // O(1) hash table lookup to get candidate positions
        let candidates = self.match_table.get(&prefix)?;

        let max_match_len = remaining.min(MAX_MATCH_LENGTH);
        let mut best_match: Option<Match> = None;
        let search_limit = self.history.len().min(MAX_MATCH_DISTANCE);

        // Check candidates in reverse order (most recent first)
        // Limit to MAX_CANDIDATES to bound worst-case performance
        for &hist_pos in candidates.iter().rev().take(MAX_CANDIDATES) {
            let distance = self.history.len() - hist_pos;

            // Skip if outside search limit
            if distance > search_limit {
                continue;
            }

            // We already know first 3 bytes match (that's why we're here)
            // Start checking from byte 3
            let mut match_len = MIN_MATCH_LENGTH;

            while match_len < max_match_len
                && hist_pos + match_len < self.history.len()
                && self.history[hist_pos + match_len] == input[pos + match_len]
            {
                match_len += 1;
            }

            // Update best match if this is longer
            if let Some(ref current_best) = best_match {
                if match_len > current_best.length {
                    best_match = Some(Match { distance, length: match_len });
                }
            } else {
                best_match = Some(Match { distance, length: match_len });
            }

            // Early exit optimization: stop if we found a very good match
            // (diminishing returns beyond this point)
            if match_len >= 32 {
                break;
            }
        }

        best_match
    }

    /// Find the appropriate match token for a given distance
    fn find_match_token(distance: usize) -> MatchToken {
        // Match tokens are entries 26-39 in TOKEN_TABLE
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

        // If distance exceeds all ranges, use last token
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

    /// Find literal token index for a byte value
    fn find_literal_token(byte: u8) -> Option<usize> {
        // Literal tokens are entries 1-25 in TOKEN_TABLE
        for (i, token) in TOKEN_TABLE.iter().enumerate().take(26).skip(1) {
            if let super::TokenType::Literal { literal_value } = token.ty {
                if literal_value == byte {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Encode a literal byte
    fn encode_literal(&self, writer: &mut BitWriter, byte: u8) -> Result<(), ZgfxError> {
        // Check if there's a literal token for this byte
        if let Some(token_idx) = Self::find_literal_token(byte) {
            let token = &TOKEN_TABLE[token_idx];
            writer.write_bits_from_slice(token.prefix);
        } else {
            // Use null literal: "0" + 8 bits
            writer.write_bit(false);
            writer.write_bits(byte as u32, 8);
        }
        Ok(())
    }

    /// Encode a match (back-reference)
    fn encode_match(&self, writer: &mut BitWriter, distance: usize, length: usize) -> Result<(), ZgfxError> {
        // Find and encode the distance token
        let match_token = Self::find_match_token(distance);

        // Write match token prefix
        writer.write_bits_from_slice(match_token.prefix);

        // Write distance value
        let distance_value = distance - match_token.distance_base;
        writer.write_bits(distance_value as u32, match_token.distance_value_size);

        // Encode length
        self.encode_match_length(writer, length)?;

        Ok(())
    }

    /// Encode match length
    fn encode_match_length(&self, writer: &mut BitWriter, length: usize) -> Result<(), ZgfxError> {
        if length == 3 {
            // Special case: single "0" bit
            writer.write_bit(false);
        } else {
            // Calculate token_size from length
            // length = base + value, where base = 2^(token_size+1)
            let length_token_size = (length as f64).log2().floor() as usize - 1;
            let base = 1 << (length_token_size + 1);
            let value = length - base;

            // Write token_size "1" bits + one "0" bit
            for _ in 0..length_token_size {
                writer.write_bit(true);
            }
            writer.write_bit(false);

            // Write value bits
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

/// Match found in history buffer
#[derive(Debug, Clone, Copy)]
struct Match {
    distance: usize,
    length: usize,
}

/// Match token information
struct MatchToken {
    prefix: &'static BitSlice<u8, Msb0>,
    distance_value_size: usize,
    distance_base: usize,
}

/// Bit-level writer for ZGFX token encoding
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

    /// Write a single bit (MSB first)
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

    /// Write multiple bits from a u32 (MSB first)
    fn write_bits(&mut self, value: u32, num_bits: usize) {
        for i in (0..num_bits).rev() {
            let bit = (value >> i) & 1 == 1;
            self.write_bit(bit);
        }
    }

    /// Write bits from a BitSlice
    fn write_bits_from_slice(&mut self, bits: &BitSlice<u8, Msb0>) {
        for bit in bits {
            self.write_bit(*bit);
        }
    }

    /// Finish writing and return bytes with padding indicator
    fn finish(mut self) -> Vec<u8> {
        // Calculate unused bits in the last byte
        let unused_bits = if self.bits_in_current == 0 {
            0
        } else {
            8 - self.bits_in_current
        };

        // Flush current byte if it has any bits
        if self.bits_in_current > 0 {
            self.bytes.push(self.current_byte);
        }

        // Add padding byte indicating unused bits
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

        // Should just be padding byte
        assert_eq!(compressed.len(), 1);
        assert_eq!(compressed[0], 0); // No unused bits
    }

    #[test]
    fn test_compress_single_byte() {
        let mut compressor = Compressor::new();
        let compressed = compressor.compress(&[0x42]).unwrap();

        // Should be encoded as null literal: "0" + 8 bits + padding
        // Total: 9 bits = 2 bytes + padding byte
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

        // Data with repetition should compress better using matches
        let data = b"AAAAAAAAAABBBBBBBBBBCCCCCCCCCC";
        let compressed = compressor.compress(data).unwrap();

        let mut output = Vec::new();
        decompressor.decompress_segment(&compressed, &mut output).unwrap();

        assert_eq!(&output, data);

        // Should achieve some compression
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

        // Large data with patterns
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

        // Write some bits
        writer.write_bit(true); // 1
        writer.write_bit(false); // 0
        writer.write_bit(true); // 1
        writer.write_bits(0b101, 3); // 101

        // Should be: 10110100 + padding
        let result = writer.finish();
        assert_eq!(result.len(), 2); // 1 data byte + 1 padding byte
        assert_eq!(result[0], 0b10110100);
        assert_eq!(result[1], 2); // 2 unused bits
    }

    #[test]
    fn test_encode_literal_with_token() {
        let compressor = Compressor::new();
        let mut writer = BitWriter::new();

        // Encode 0x00 which has a literal token (index 1: 11000)
        compressor.encode_literal(&mut writer, 0x00).unwrap();

        let result = writer.finish();
        // Should start with 11000...
        assert!(!result.is_empty());
    }

    #[test]
    fn test_encode_literal_null() {
        let compressor = Compressor::new();
        let mut writer = BitWriter::new();

        // Encode a byte without a literal token (e.g., 0x42)
        compressor.encode_literal(&mut writer, 0x42).unwrap();

        let result = writer.finish();
        // Should be: 0 + 01000010 (0x42) + padding = 9 bits
        // = 2 data bytes + 1 padding indicator byte = 3 total
        assert_eq!(result.len(), 3);
        // First byte should be 00100001 (bits 0-7)
        // Second byte should be 00000000 (bit 8 plus padding)
        assert_eq!(result[2], 7); // 7 unused bits in last byte
    }
}
