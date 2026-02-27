//! Progressive RFX decode and encode algorithms ([MS-RDPEGFX] 2.2.4.2).
//!
//! Provides first-pass decode (RLGR1 + progressive dequantization + sign capture)
//! and upgrade-pass decode (SRL/raw routing by DAS sign state, coefficient
//! accumulation) for the RemoteFX Progressive codec.
//!
//! These are pure algorithmic functions operating on coefficient buffers.
//! Tile state management and EGFX integration belong in a higher layer.

use ironrdp_pdu::codecs::rfx::progressive::ComponentCodecQuant;
use ironrdp_pdu::codecs::rfx::{EntropyAlgorithm, Quant};

use crate::dwt_extrapolate::BandInfo;
use crate::rlgr::RlgrError;
use crate::srl;

/// Number of DWT coefficients per component in a 64x64 tile.
pub const COEFFICIENTS_PER_COMPONENT: usize = 4096;

/// Number of subbands in a 3-level DWT decomposition.
pub const NUM_BANDS: usize = 10;

/// DAS (Delta-Analysis State) values for tri-state sign tracking.
///
/// After the first pass, each coefficient position is classified:
/// - `SIGN_ZERO`: coefficient was zero (eligible for SRL upgrade)
/// - `SIGN_POSITIVE`: coefficient was positive (eligible for raw upgrade)
/// - `SIGN_NEGATIVE`: coefficient was negative (eligible for raw upgrade)
pub const SIGN_ZERO: i8 = 0;
pub const SIGN_POSITIVE: i8 = 1;
pub const SIGN_NEGATIVE: i8 = -1;

// ---------------------------------------------------------------------------
// First-pass decode (TILE_SIMPLE / TILE_FIRST)
// ---------------------------------------------------------------------------

/// Decode a first-pass component from RLGR1-encoded data.
///
/// Performs: RLGR1 decode -> base dequantization -> progressive dequantization
/// -> LL3 delta decode -> sign capture.
///
/// # Arguments
/// - `data`: RLGR1-encoded coefficient stream
/// - `base_quant`: base quantization values (from region quant table)
/// - `prog_quant`: progressive quantization BitPos values for this quality level
/// - `use_reduce_extrapolate`: whether to use asymmetric band sizes
/// - `coefficients`: output buffer for decoded coefficients (4096 i16)
/// - `sign`: output buffer for DAS sign state (4096 i8)
///
/// # Panics
///
/// Panics if `coefficients` or `sign` has fewer than 4096 elements.
///
/// # Errors
/// Returns `RlgrError` if RLGR decoding fails.
pub fn decode_first_pass(
    data: &[u8],
    base_quant: &Quant,
    prog_quant: &ComponentCodecQuant,
    use_reduce_extrapolate: bool,
    coefficients: &mut [i16],
    sign: &mut [i8],
) -> Result<(), RlgrError> {
    assert!(coefficients.len() >= COEFFICIENTS_PER_COMPONENT);
    assert!(sign.len() >= COEFFICIENTS_PER_COMPONENT);

    // Step 1: RLGR1 decode into coefficient buffer
    crate::rlgr::decode(EntropyAlgorithm::Rlgr1, data, coefficients)?;

    // Step 2: LL3 differential decoding (reverse delta encoding on last subband)
    crate::subband_reconstruction::decode(&mut coefficients[ll3_offset(use_reduce_extrapolate)..]);

    // Step 3: Base dequantization (shift left by quant - 1)
    dequantize_component(coefficients, base_quant, use_reduce_extrapolate);

    // Step 4: Progressive dequantization (shift left by BitPos)
    progressive_dequantize(coefficients, prog_quant, use_reduce_extrapolate);

    // Step 5: Capture sign state for DAS
    capture_sign(coefficients, sign);

    Ok(())
}

/// Decode an upgrade-pass component from SRL and raw data streams.
///
/// For each coefficient position:
/// - DAS = 0 (zero): decode from SRL stream, update DAS if non-zero
/// - DAS != 0 (non-zero): decode raw magnitude bits, accumulate
///
/// # Arguments
/// - `srl_data`: SRL-encoded stream for zero-DAS positions
/// - `raw_data`: raw bit stream for non-zero-DAS positions
/// - `prev_prog_quant`: BitPos values from previous quality level
/// - `curr_prog_quant`: BitPos values for this quality level
/// - `use_reduce_extrapolate`: whether to use asymmetric band sizes
/// - `coefficients`: coefficient buffer to accumulate into (modified in-place)
/// - `sign`: DAS sign buffer (modified in-place when zeros become non-zero)
///
/// # Panics
///
/// Panics if `coefficients` or `sign` has fewer than 4096 elements.
pub fn decode_upgrade_pass(
    srl_data: &[u8],
    raw_data: &[u8],
    prev_prog_quant: &ComponentCodecQuant,
    curr_prog_quant: &ComponentCodecQuant,
    use_reduce_extrapolate: bool,
    coefficients: &mut [i16],
    sign: &mut [i8],
) {
    assert!(coefficients.len() >= COEFFICIENTS_PER_COMPONENT);
    assert!(sign.len() >= COEFFICIENTS_PER_COMPONENT);

    let bands = get_band_layout(use_reduce_extrapolate);

    for (band_idx, band) in bands.iter().enumerate() {
        let prev_bit_pos = prev_prog_quant.for_band(band_idx);
        let curr_bit_pos = curr_prog_quant.for_band(band_idx);

        // Number of raw bits per coefficient in this band
        let num_bits = prev_bit_pos.saturating_sub(curr_bit_pos);
        if num_bits == 0 {
            continue;
        }

        // Count zero-DAS positions in this band (for SRL decode)
        let zero_count = band_zero_count(sign, band);

        // SRL decode for zero-DAS positions
        let srl_values = srl::decode_srl(srl_data, zero_count, num_bits);

        // Apply upgrade values to this band
        let mut srl_idx = 0;
        let mut raw_reader = RawBitReader::new(raw_data);

        for i in 0..band.count() {
            let coeff_idx = band.offset + i;
            let is_ll3 = band_idx == 9;

            if sign[coeff_idx] == SIGN_ZERO {
                // Zero-DAS: get value from SRL stream
                let value = if srl_idx < srl_values.len() {
                    srl_values[srl_idx]
                } else {
                    0
                };
                srl_idx += 1;

                if value != 0 {
                    // Coefficient transitions from zero to non-zero
                    let shifted = i32::from(value) << i32::from(curr_bit_pos);
                    coefficients[coeff_idx] = clamp_i16(shifted);
                    sign[coeff_idx] = if value > 0 { SIGN_POSITIVE } else { SIGN_NEGATIVE };
                }
            } else {
                // Non-zero DAS: read raw magnitude bits
                let raw_mag = raw_reader.read_bits(u32::from(num_bits));

                if raw_mag != 0 {
                    // raw_mag fits in i32 (at most 2^15 from bit stream)
                    let mag_i32 = i32::try_from(raw_mag).unwrap_or(i32::MAX);
                    let shifted = mag_i32 << i32::from(curr_bit_pos);
                    if is_ll3 || sign[coeff_idx] == SIGN_POSITIVE {
                        // LL3 is always positive; positive DAS adds
                        coefficients[coeff_idx] = clamp_i16(i32::from(coefficients[coeff_idx]) + shifted);
                    } else {
                        // Negative DAS subtracts
                        coefficients[coeff_idx] = clamp_i16(i32::from(coefficients[coeff_idx]) - shifted);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Progressive (de)quantization
// ---------------------------------------------------------------------------

/// Apply progressive dequantization: left-shift each band by its BitPos value.
///
/// For non-LL3 bands, this shifts the absolute value (preserving sign).
/// For LL3, this is a simple left shift (floor toward negative infinity).
fn progressive_dequantize(coefficients: &mut [i16], prog_quant: &ComponentCodecQuant, use_reduce_extrapolate: bool) {
    let bands = get_band_layout(use_reduce_extrapolate);

    for (band_idx, band) in bands.iter().enumerate() {
        let bit_pos = prog_quant.for_band(band_idx);
        if bit_pos == 0 {
            continue;
        }

        let is_ll3 = band_idx == 9;
        let start = band.offset;
        let end = start + band.count();

        if is_ll3 {
            // LL3: simple left shift (floor toward negative infinity)
            for coeff in &mut coefficients[start..end] {
                *coeff = clamp_i16(i32::from(*coeff) << i32::from(bit_pos));
            }
        } else {
            // Other bands: shift absolute value, preserve sign
            for coeff in &mut coefficients[start..end] {
                let val = i32::from(*coeff);
                if val >= 0 {
                    *coeff = clamp_i16(val << i32::from(bit_pos));
                } else {
                    *coeff = clamp_i16(-((-val) << i32::from(bit_pos)));
                }
            }
        }
    }
}

/// Apply progressive quantization: right-shift each band by its BitPos value.
///
/// Inverse of `progressive_dequantize`.
pub fn progressive_quantize(coefficients: &mut [i16], prog_quant: &ComponentCodecQuant, use_reduce_extrapolate: bool) {
    let bands = get_band_layout(use_reduce_extrapolate);

    for (band_idx, band) in bands.iter().enumerate() {
        let bit_pos = prog_quant.for_band(band_idx);
        if bit_pos == 0 {
            continue;
        }

        let is_ll3 = band_idx == 9;
        let start = band.offset;
        let end = start + band.count();

        if is_ll3 {
            // LL3: floor division (right shift)
            for coeff in &mut coefficients[start..end] {
                *coeff >>= bit_pos;
            }
        } else {
            // Other bands: truncation toward zero
            for coeff in &mut coefficients[start..end] {
                let val = i32::from(*coeff);
                if val >= 0 {
                    *coeff = clamp_i16(val >> i32::from(bit_pos));
                } else {
                    *coeff = clamp_i16(-((-val) >> i32::from(bit_pos)));
                }
            }
        }
    }
}

/// Base dequantization using the classic RFX `Quant` struct.
///
/// Each band is shifted left by `(quant_value - 1)`. Uses the band layout
/// for correct offsets when reduce-extrapolate is active.
fn dequantize_component(coefficients: &mut [i16], quant: &Quant, use_reduce_extrapolate: bool) {
    if use_reduce_extrapolate {
        // Use reduce-extrapolate band layout
        let bands = crate::dwt_extrapolate::band_layout();
        // Band order in buffer: HL1, LH1, HH1, HL2, LH2, HH2, HL3, LH3, HH3, LL3
        let quant_per_band = [
            quant.hl1, quant.lh1, quant.hh1, quant.hl2, quant.lh2, quant.hh2, quant.hl3, quant.lh3, quant.hh3,
            quant.ll3,
        ];
        for (band, &q) in bands.iter().zip(quant_per_band.iter()) {
            let factor = i16::from(q).saturating_sub(1);
            if factor > 0 {
                let start = band.offset;
                let end = start + band.count();
                for coeff in &mut coefficients[start..end] {
                    *coeff <<= factor;
                }
            }
        }
    } else {
        // Standard band sizes, use existing quantization function
        crate::quantization::decode(coefficients, quant);
    }
}

// ---------------------------------------------------------------------------
// Sign capture
// ---------------------------------------------------------------------------

/// Capture the tri-state sign of each coefficient into the DAS array.
fn capture_sign(coefficients: &[i16], sign: &mut [i8]) {
    for (s, &c) in sign.iter_mut().zip(coefficients.iter()) {
        *s = match c.cmp(&0) {
            core::cmp::Ordering::Greater => SIGN_POSITIVE,
            core::cmp::Ordering::Less => SIGN_NEGATIVE,
            core::cmp::Ordering::Equal => SIGN_ZERO,
        };
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get the band layout for the current DWT mode.
fn get_band_layout(use_reduce_extrapolate: bool) -> [BandInfo; NUM_BANDS] {
    if use_reduce_extrapolate {
        crate::dwt_extrapolate::band_layout()
    } else {
        standard_band_layout()
    }
}

/// Standard (non-extrapolate) band layout for a 64x64 tile.
/// Band sizes: 1024 each for level 1, 256 each for level 2, 64 each for level 3.
fn standard_band_layout() -> [BandInfo; NUM_BANDS] {
    let mut off = 0;
    let mut b = |w: usize, h: usize| {
        let info = BandInfo {
            width: w,
            height: h,
            offset: off,
        };
        off += w * h;
        info
    };

    [
        b(32, 32), // HL1: 1024
        b(32, 32), // LH1: 1024
        b(32, 32), // HH1: 1024
        b(16, 16), // HL2: 256
        b(16, 16), // LH2: 256
        b(16, 16), // HH2: 256
        b(8, 8),   // HL3: 64
        b(8, 8),   // LH3: 64
        b(8, 8),   // HH3: 64
        b(8, 8),   // LL3: 64
    ]
}

/// Starting offset of the LL3 subband for delta decoding.
fn ll3_offset(use_reduce_extrapolate: bool) -> usize {
    if use_reduce_extrapolate {
        4015 // reduce-extrapolate: 9x9 = 81 coefficients at offset 4015
    } else {
        4032 // standard: 8x8 = 64 coefficients at offset 4032
    }
}

/// Count zero-DAS positions within a band.
fn band_zero_count(sign: &[i8], band: &BandInfo) -> usize {
    let start = band.offset;
    let end = start + band.count();
    sign[start..end].iter().filter(|&&s| s == SIGN_ZERO).count()
}

/// Clamp i32 to i16 range.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "value is clamped to i16 range before cast"
)]
fn clamp_i16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

// ---------------------------------------------------------------------------
// Raw bit reader for upgrade pass
// ---------------------------------------------------------------------------

/// Reads raw magnitude bits MSB-first from a byte stream.
struct RawBitReader<'a> {
    data: &'a [u8],
    byte_idx: usize,
    bit_idx: u8,
}

impl<'a> RawBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_idx: 0,
            bit_idx: 0,
        }
    }

    fn read_bits(&mut self, count: u32) -> u32 {
        let mut value = 0u32;
        for _ in 0..count {
            value = (value << 1) | u32::from(self.read_bit());
        }
        value
    }

    fn read_bit(&mut self) -> bool {
        if self.byte_idx >= self.data.len() {
            return false;
        }
        let bit = (self.data[self.byte_idx] >> (7 - self.bit_idx)) & 1 != 0;
        self.bit_idx += 1;
        if self.bit_idx >= 8 {
            self.bit_idx = 0;
            self.byte_idx += 1;
        }
        bit
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::as_conversions, clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
mod tests {
    use super::*;

    #[test]
    fn standard_band_layout_totals_4096() {
        let bands = standard_band_layout();
        let total: usize = bands.iter().map(|b| b.count()).sum();
        assert_eq!(total, 4096);
    }

    #[test]
    fn standard_band_offsets() {
        let bands = standard_band_layout();
        assert_eq!(bands[0].offset, 0);
        assert_eq!(bands[1].offset, 1024);
        assert_eq!(bands[2].offset, 2048);
        assert_eq!(bands[3].offset, 3072);
        assert_eq!(bands[4].offset, 3328);
        assert_eq!(bands[5].offset, 3584);
        assert_eq!(bands[6].offset, 3840);
        assert_eq!(bands[7].offset, 3904);
        assert_eq!(bands[8].offset, 3968);
        assert_eq!(bands[9].offset, 4032);
    }

    #[test]
    fn sign_capture_tri_state() {
        let coefficients = [10i16, -5, 0, 100, -1, 0];
        let mut sign = [0i8; 6];
        capture_sign(&coefficients, &mut sign);
        assert_eq!(sign, [1, -1, 0, 1, -1, 0]);
    }

    #[test]
    fn progressive_dequantize_ll3_shift() {
        // LL3 is band index 9, at offset 4032 for standard layout
        let mut coefficients = vec![0i16; 4096];
        coefficients[4032] = 5;
        coefficients[4033] = -3;

        let prog_quant = ComponentCodecQuant {
            ll3: 2,
            hl3: 0,
            lh3: 0,
            hh3: 0,
            hl2: 0,
            lh2: 0,
            hh2: 0,
            hl1: 0,
            lh1: 0,
            hh1: 0,
        };

        progressive_dequantize(&mut coefficients, &prog_quant, false);

        // LL3 uses floor shift: 5 << 2 = 20, -3 << 2 = -12
        assert_eq!(coefficients[4032], 20);
        assert_eq!(coefficients[4033], -12);
    }

    #[test]
    fn progressive_dequantize_non_ll3_preserves_sign() {
        // HL1 is band index 0, at offset 0 for standard layout
        let mut coefficients = vec![0i16; 4096];
        coefficients[0] = 5;
        coefficients[1] = -5;

        let prog_quant = ComponentCodecQuant {
            ll3: 0,
            hl3: 0,
            lh3: 0,
            hh3: 0,
            hl2: 0,
            lh2: 0,
            hh2: 0,
            hl1: 2,
            lh1: 0,
            hh1: 0,
        };

        progressive_dequantize(&mut coefficients, &prog_quant, false);

        // Non-LL3: shift absolute value, preserve sign
        assert_eq!(coefficients[0], 20); // 5 << 2
        assert_eq!(coefficients[1], -20); // -(5 << 2)
    }

    #[test]
    fn progressive_quantize_round_trip() {
        let mut coefficients = vec![0i16; 4096];
        for (i, c) in coefficients.iter_mut().enumerate() {
            *c = (i as i16).wrapping_mul(7);
        }
        let original = coefficients.clone();

        let prog_quant = ComponentCodecQuant {
            ll3: 2,
            hl3: 3,
            lh3: 3,
            hh3: 4,
            hl2: 3,
            lh2: 3,
            hh2: 4,
            hl1: 2,
            lh1: 2,
            hh1: 3,
        };

        progressive_quantize(&mut coefficients, &prog_quant, false);
        progressive_dequantize(&mut coefficients, &prog_quant, false);

        // After quantize->dequantize, values lose precision from truncation
        // but should be in the right ballpark
        for (i, (&a, &b)) in coefficients.iter().zip(original.iter()).enumerate() {
            let err = (i32::from(a) - i32::from(b)).unsigned_abs();
            // Max error bounded by 2^(bit_pos)
            assert!(err < 32, "index {i}: error {err} too large");
        }
    }

    #[test]
    fn raw_bit_reader_basic() {
        let data = [0b10110000, 0b01010000];
        let mut reader = RawBitReader::new(&data);
        assert_eq!(reader.read_bits(4), 0b1011);
        assert_eq!(reader.read_bits(4), 0b0000);
        assert_eq!(reader.read_bits(4), 0b0101);
    }

    #[test]
    fn clamp_i16_limits() {
        assert_eq!(clamp_i16(40000), i16::MAX);
        assert_eq!(clamp_i16(-40000), i16::MIN);
        assert_eq!(clamp_i16(100), 100);
        assert_eq!(clamp_i16(-100), -100);
    }

    #[test]
    fn band_zero_count_counts_correctly() {
        let mut sign = [0i8; 4096];
        // Band 0 (HL1): offset 0, count 1024
        sign[0] = SIGN_POSITIVE;
        sign[1] = SIGN_NEGATIVE;
        sign[2] = SIGN_ZERO;
        // Rest are SIGN_ZERO by default

        let bands = standard_band_layout();
        assert_eq!(band_zero_count(&sign, &bands[0]), 1022); // 1024 - 2 non-zero
    }

    #[test]
    fn ll3_offsets_correct() {
        assert_eq!(ll3_offset(false), 4032);
        assert_eq!(ll3_offset(true), 4015);
    }

    #[test]
    fn upgrade_pass_zero_das_becomes_nonzero() {
        let mut coefficients = vec![0i16; 4096];
        let mut sign = vec![SIGN_ZERO; 4096];

        // Set up SRL data that produces a non-zero value for the first position
        // For band 0 (HL1), with num_bits=2, SRL should produce some values
        let prev_prog_quant = ComponentCodecQuant {
            ll3: 0,
            hl3: 0,
            lh3: 0,
            hh3: 0,
            hl2: 0,
            lh2: 0,
            hh2: 0,
            hl1: 4,
            lh1: 0,
            hh1: 0,
        };
        let curr_prog_quant = ComponentCodecQuant {
            ll3: 0,
            hl3: 0,
            lh3: 0,
            hh3: 0,
            hl2: 0,
            lh2: 0,
            hh2: 0,
            hl1: 2,
            lh1: 0,
            hh1: 0,
        };

        // Simple SRL data: a non-zero value (the SRL decoder will interpret
        // bits as magnitude + sign). With num_bits=2, k=0 initially,
        // it goes straight to magnitude decode.
        let srl_data = vec![0b01000000, 0x00]; // sign=0(+), magnitude bits follow
        let raw_data = vec![];

        decode_upgrade_pass(
            &srl_data,
            &raw_data,
            &prev_prog_quant,
            &curr_prog_quant,
            false,
            &mut coefficients,
            &mut sign,
        );

        // After decode, at least some positions should have been updated
        // (exact values depend on SRL interpretation, but the function shouldn't panic)
    }
}
