//! V2 acknowledgment payloads.
//!
//! `Acknowledgement Payload` (MS-RDPEUDP2 Section 2.2.1.2.1) and
//! `Acknowledgement Vector Payload` (MS-RDPEUDP2 Section 2.2.1.2.6).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use ironrdp_core::{Decode, DecodeResult, Encode, EncodeResult, ReadCursor, WriteCursor};

// ── Acknowledgement Payload (ACK flag) ──

/// Acknowledges one or more consecutively received packets.
///
/// MS-RDPEUDP2 Section 2.2.1.2.1.
/// Present when the ACK flag (0x001) is set in the header.
///
/// Wire layout:
/// - `SeqNum(2)` + `receivedTS(3)` + `sendAckTimeGap(1)` +
///   `numDelayedAcks(4 bits) : delayAckTimeScale(4 bits)` = 7 bytes fixed.
/// - `delayAckTimeAdditions(numDelayedAcks bytes)` = variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AckPayload {
    /// Lower 16 bits of the acknowledged data packet's sequence number.
    pub seq_num: u16,

    /// Lower 24 bits of the timestamp when the packet was received.
    /// Units of 4 microseconds.
    pub received_ts: u32,

    /// Time in milliseconds between packet receipt and ACK send.
    pub send_ack_time_gap: u8,

    /// Number of additional delayed ACKs bundled (0..=15).
    pub num_delayed_acks: u8,

    /// Scale for delay time additions (0..=15).
    /// Each addition is in units of `(1 << delay_ack_time_scale)` microseconds.
    pub delay_ack_time_scale: u8,

    /// Per-ACK time deltas in reverse chronological order.
    /// Length = `num_delayed_acks`.
    /// Each byte × `(1 << delay_ack_time_scale)` microseconds gives the
    /// time between adjacent acknowledgments.
    pub delay_ack_time_additions: Vec<u8>,
}

impl AckPayload {
    /// Fixed part size: SeqNum(2) + receivedTS(3) + sendAckTimeGap(1) + packed_nibbles(1) = 7.
    const FIXED_PART_SIZE: usize = 7;
    const NAME: &'static str = "RDP-UDP2 Acknowledgement Payload";

    /// Mask for the lower 24 bits of a timestamp.
    pub const TIMESTAMP_MASK: u32 = 0x00FF_FFFF;
}

impl Encode for AckPayload {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_size!(in: dst, size: self.size());

        dst.write_u16(self.seq_num);

        // receivedTS: 3 bytes little-endian (lower 24 bits)
        let ts_bytes = (self.received_ts & Self::TIMESTAMP_MASK).to_le_bytes();
        dst.write_u8(ts_bytes[0]);
        dst.write_u8(ts_bytes[1]);
        dst.write_u8(ts_bytes[2]);

        dst.write_u8(self.send_ack_time_gap);

        // Pack numDelayedAcks(high nibble) and delayAckTimeScale(low nibble)
        let packed = ((self.num_delayed_acks & 0x0F) << 4) | (self.delay_ack_time_scale & 0x0F);
        dst.write_u8(packed);

        dst.write_slice(&self.delay_ack_time_additions);

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE + self.delay_ack_time_additions.len()
    }
}

impl Decode<'_> for AckPayload {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_fixed_part_size!(in: src);

        let seq_num = src.read_u16();

        // receivedTS: 3 bytes little-endian
        let ts_b0 = u32::from(src.read_u8());
        let ts_b1 = u32::from(src.read_u8());
        let ts_b2 = u32::from(src.read_u8());
        let received_ts = ts_b0 | (ts_b1 << 8) | (ts_b2 << 16);

        let send_ack_time_gap = src.read_u8();

        let packed = src.read_u8();
        let num_delayed_acks = (packed >> 4) & 0x0F;
        let delay_ack_time_scale = packed & 0x0F;

        let additions_len = usize::from(num_delayed_acks);
        ironrdp_core::ensure_size!(in: src, size: additions_len);
        let delay_ack_time_additions = src.read_slice(additions_len).to_vec();

        Ok(Self {
            seq_num,
            received_ts,
            send_ack_time_gap,
            num_delayed_acks,
            delay_ack_time_scale,
            delay_ack_time_additions,
        })
    }
}

// ── Acknowledgement Vector Payload (ACKVEC flag) ──

/// Per-packet acknowledgment state encoding mode.
///
/// MS-RDPEUDP2 Section 2.2.1.2.6.
/// Each byte in the coded ACK vector is independently encoded as either
/// a state-map (MSB = 0) or a run-length (MSB = 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AckVectorEntry {
    /// State-map mode (MSB = 0).
    /// Remaining 7 bits are a bitmap: each bit represents one sequence number
    /// (1 = received, 0 = not received), from lowest to highest bit.
    StateMap {
        /// 7-bit bitmap of received/not-received states.
        bitmap: u8,
    },

    /// Run-length mode (MSB = 1).
    /// Bit 6 = state (1 = received, 0 = not received).
    /// Bits 0-5 = run length (number of consecutive packets in this state).
    RunLength {
        /// Whether the packets in this run were received.
        received: bool,
        /// Number of consecutive packets in this state (0..=63).
        length: u8,
    },
}

impl AckVectorEntry {
    /// Encode to a single byte.
    pub fn to_byte(self) -> u8 {
        match self {
            Self::StateMap { bitmap } => bitmap & 0x7F, // MSB = 0
            Self::RunLength { received, length } => {
                let mut byte = 0x80; // MSB = 1
                if received {
                    byte |= 0x40; // bit 6 = state
                }
                byte | (length & 0x3F)
            }
        }
    }

    /// Decode from a single byte.
    pub fn from_byte(byte: u8) -> Self {
        if (byte & 0x80) == 0 {
            // State-map mode
            Self::StateMap { bitmap: byte & 0x7F }
        } else {
            // Run-length mode
            Self::RunLength {
                received: (byte & 0x40) != 0,
                length: byte & 0x3F,
            }
        }
    }

    /// Number of sequence numbers covered by this entry.
    pub fn coverage(self) -> u16 {
        match self {
            Self::StateMap { .. } => 7,
            Self::RunLength { length, .. } => u16::from(length),
        }
    }
}

/// Vector of per-packet acknowledgment states within the receiver window.
///
/// MS-RDPEUDP2 Section 2.2.1.2.6.
/// Present when the ACKVEC flag (0x008) is set in the header.
/// Mutually exclusive with the ACK payload.
///
/// Wire layout:
/// - `BaseSeqNum(2)` + packed byte (`codedAckVecSize(7 bits) : TimeStampPresent(1 bit)`) = 3 bytes.
/// - If `TimeStampPresent`: `SendAckTimeGapInMs(1)` + `TimeStamp(3)` = 4 bytes.
/// - `codedAckVector(codedAckVecSize bytes)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AckVectorPayload {
    /// Lower 16 bits of the base sequence number for this vector.
    pub base_seq_num: u16,

    /// Lower 24 bits of timestamp for the highest unacked sequence number received.
    /// Units of 4 microseconds. Present only when `timestamp_present`.
    pub timestamp: Option<u32>,

    /// Time in ms between ACK send and last data packet receipt.
    /// 255 = invalid/unused. Present only when `timestamp_present`.
    pub send_ack_time_gap_ms: Option<u8>,

    /// Encoded ACK vector entries.
    pub entries: Vec<AckVectorEntry>,
}

impl AckVectorPayload {
    const NAME: &'static str = "RDP-UDP2 Acknowledgement Vector Payload";

    /// Minimum fixed part: BaseSeqNum(2) + packed_byte(1) = 3 bytes.
    const MIN_FIXED_SIZE: usize = 3;

    /// Optional timestamp block: SendAckTimeGapInMs(1) + TimeStamp(3) = 4 bytes.
    const TIMESTAMP_BLOCK_SIZE: usize = 4;
}

impl Encode for AckVectorPayload {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_size!(in: dst, size: self.size());

        dst.write_u16(self.base_seq_num);

        let vec_size: u8 = ironrdp_core::cast_length!("AckVectorPayload", "codedAckVecSize", self.entries.len())?;
        if vec_size > 0x7F {
            return Err(ironrdp_core::invalid_field_err!(
                "AckVectorPayload",
                "codedAckVecSize",
                "exceeds 7-bit maximum (127)"
            ));
        }

        let timestamp_present = self.timestamp.is_some();
        let packed = (vec_size & 0x7F) | if timestamp_present { 0x80 } else { 0x00 };
        dst.write_u8(packed);

        if let (Some(ts), Some(gap)) = (self.timestamp, self.send_ack_time_gap_ms) {
            dst.write_u8(gap);
            // TimeStamp: 3 bytes little-endian
            let ts_bytes = (ts & AckPayload::TIMESTAMP_MASK).to_le_bytes();
            dst.write_u8(ts_bytes[0]);
            dst.write_u8(ts_bytes[1]);
            dst.write_u8(ts_bytes[2]);
        }

        for entry in &self.entries {
            dst.write_u8(entry.to_byte());
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        let mut total = Self::MIN_FIXED_SIZE;
        if self.timestamp.is_some() {
            total += Self::TIMESTAMP_BLOCK_SIZE;
        }
        total += self.entries.len();
        total
    }
}

impl Decode<'_> for AckVectorPayload {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_size!(in: src, size: Self::MIN_FIXED_SIZE);

        let base_seq_num = src.read_u16();

        let packed = src.read_u8();
        let coded_ack_vec_size = packed & 0x7F;
        let timestamp_present = (packed & 0x80) != 0;

        let (timestamp, send_ack_time_gap_ms) = if timestamp_present {
            ironrdp_core::ensure_size!(in: src, size: Self::TIMESTAMP_BLOCK_SIZE);
            let gap = src.read_u8();
            let ts_b0 = u32::from(src.read_u8());
            let ts_b1 = u32::from(src.read_u8());
            let ts_b2 = u32::from(src.read_u8());
            let ts = ts_b0 | (ts_b1 << 8) | (ts_b2 << 16);
            (Some(ts), Some(gap))
        } else {
            (None, None)
        };

        let entries_len = usize::from(coded_ack_vec_size);
        ironrdp_core::ensure_size!(in: src, size: entries_len);
        let mut entries = Vec::with_capacity(entries_len);
        for _ in 0..entries_len {
            entries.push(AckVectorEntry::from_byte(src.read_u8()));
        }

        Ok(Self {
            base_seq_num,
            timestamp,
            send_ack_time_gap_ms,
            entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use ironrdp_core::{decode, encode_vec};

    use super::*;

    // ── AckPayload tests ──

    #[test]
    fn ack_payload_no_delayed() {
        let ack = AckPayload {
            seq_num: 0x1234,
            received_ts: 0x00AB_CDEF,
            send_ack_time_gap: 5,
            num_delayed_acks: 0,
            delay_ack_time_scale: 0,
            delay_ack_time_additions: Vec::new(),
        };
        let encoded = encode_vec(&ack).expect("encode");
        assert_eq!(ack.size(), 7);
        assert_eq!(encoded.len(), 7);

        let decoded: AckPayload = decode(&encoded).expect("decode");
        assert_eq!(decoded, ack);
    }

    #[test]
    fn ack_payload_with_delayed() {
        let ack = AckPayload {
            seq_num: 0x0042,
            received_ts: 0x00_123456,
            send_ack_time_gap: 10,
            num_delayed_acks: 3,
            delay_ack_time_scale: 2,
            delay_ack_time_additions: vec![15, 20, 25],
        };
        let encoded = encode_vec(&ack).expect("encode");
        assert_eq!(ack.size(), 10); // 7 + 3
        assert_eq!(encoded.len(), 10);

        let decoded: AckPayload = decode(&encoded).expect("decode");
        assert_eq!(decoded, ack);
    }

    #[test]
    fn ack_payload_timestamp_24bit() {
        let ack = AckPayload {
            seq_num: 0,
            received_ts: 0x00FF_FFFF, // max 24-bit value
            send_ack_time_gap: 0,
            num_delayed_acks: 0,
            delay_ack_time_scale: 0,
            delay_ack_time_additions: Vec::new(),
        };
        let encoded = encode_vec(&ack).expect("encode");
        let decoded: AckPayload = decode(&encoded).expect("decode");
        assert_eq!(decoded.received_ts, 0x00FF_FFFF);
    }

    #[test]
    fn ack_payload_timestamp_masked() {
        // Setting bits above 24 should be masked on encode
        let ack = AckPayload {
            seq_num: 0,
            received_ts: 0xFFFF_FFFF, // bits above 24 set
            send_ack_time_gap: 0,
            num_delayed_acks: 0,
            delay_ack_time_scale: 0,
            delay_ack_time_additions: Vec::new(),
        };
        let encoded = encode_vec(&ack).expect("encode");
        let decoded: AckPayload = decode(&encoded).expect("decode");
        assert_eq!(decoded.received_ts, 0x00FF_FFFF);
    }

    #[test]
    fn ack_payload_nibble_packing() {
        // Verify the 4-bit nibble packing
        let ack = AckPayload {
            seq_num: 0,
            received_ts: 0,
            send_ack_time_gap: 0,
            num_delayed_acks: 15,    // max for 4 bits
            delay_ack_time_scale: 8, // arbitrary 4-bit value
            delay_ack_time_additions: vec![0; 15],
        };
        let encoded = encode_vec(&ack).expect("encode");
        // The packed byte should be: (15 << 4) | 8 = 0xF8
        assert_eq!(encoded[6], 0xF8);

        let decoded: AckPayload = decode(&encoded).expect("decode");
        assert_eq!(decoded.num_delayed_acks, 15);
        assert_eq!(decoded.delay_ack_time_scale, 8);
    }

    #[test]
    fn ack_payload_insufficient_bytes() {
        let bytes = [0x00, 0x00, 0x00]; // only 3 bytes, need 7
        let result: DecodeResult<AckPayload> = decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn ack_payload_insufficient_additions() {
        // Claims 5 delayed acks but no addition bytes follow
        let bytes = [
            0x00, 0x00, // seq_num
            0x00, 0x00, 0x00, // received_ts
            0x00, // send_ack_time_gap
            0x50, // numDelayedAcks=5, scale=0
                  // missing 5 addition bytes
        ];
        let result: DecodeResult<AckPayload> = decode(&bytes);
        assert!(result.is_err());
    }

    // ── AckVectorEntry tests ──

    #[test]
    fn state_map_entry() {
        let entry = AckVectorEntry::StateMap { bitmap: 0b0110_1010 };
        let byte = entry.to_byte();
        assert_eq!(byte & 0x80, 0, "MSB should be 0 for state-map");
        assert_eq!(byte, 0b0110_1010);
        let decoded = AckVectorEntry::from_byte(byte);
        assert_eq!(decoded, entry);
    }

    #[test]
    fn run_length_received() {
        let entry = AckVectorEntry::RunLength {
            received: true,
            length: 42,
        };
        let byte = entry.to_byte();
        assert_eq!(byte, 0x80 | 0x40 | 42); // MSB=1, state=1, length=42
        let decoded = AckVectorEntry::from_byte(byte);
        assert_eq!(decoded, entry);
    }

    #[test]
    fn run_length_not_received() {
        let entry = AckVectorEntry::RunLength {
            received: false,
            length: 7,
        };
        let byte = entry.to_byte();
        assert_eq!(byte, 0x80 | 7); // MSB=1, state=0, length=7
        let decoded = AckVectorEntry::from_byte(byte);
        assert_eq!(decoded, entry);
    }

    #[test]
    fn run_length_max() {
        let entry = AckVectorEntry::RunLength {
            received: true,
            length: 63, // 6-bit max
        };
        let byte = entry.to_byte();
        assert_eq!(byte, 0xFF); // 0x80 | 0x40 | 0x3F
        let decoded = AckVectorEntry::from_byte(byte);
        assert_eq!(decoded, entry);
    }

    #[test]
    fn entry_coverage() {
        let map = AckVectorEntry::StateMap { bitmap: 0x7F };
        assert_eq!(map.coverage(), 7);

        let run = AckVectorEntry::RunLength {
            received: true,
            length: 30,
        };
        assert_eq!(run.coverage(), 30);
    }

    // ── AckVectorPayload tests ──

    #[test]
    fn ack_vector_no_timestamp() {
        let payload = AckVectorPayload {
            base_seq_num: 0x0100,
            timestamp: None,
            send_ack_time_gap_ms: None,
            entries: vec![
                AckVectorEntry::RunLength {
                    received: true,
                    length: 10,
                },
                AckVectorEntry::StateMap { bitmap: 0b0101_010 },
            ],
        };
        let encoded = encode_vec(&payload).expect("encode");
        assert_eq!(payload.size(), 3 + 2); // 3 fixed + 2 entries
        assert_eq!(encoded.len(), 5);

        let decoded: AckVectorPayload = decode(&encoded).expect("decode");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn ack_vector_with_timestamp() {
        let payload = AckVectorPayload {
            base_seq_num: 0x0042,
            timestamp: Some(0x00AB_CDEF),
            send_ack_time_gap_ms: Some(25),
            entries: vec![AckVectorEntry::RunLength {
                received: true,
                length: 20,
            }],
        };
        let encoded = encode_vec(&payload).expect("encode");
        assert_eq!(payload.size(), 3 + 4 + 1); // 3 fixed + 4 timestamp block + 1 entry
        assert_eq!(encoded.len(), 8);

        let decoded: AckVectorPayload = decode(&encoded).expect("decode");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn ack_vector_empty_entries() {
        let payload = AckVectorPayload {
            base_seq_num: 0,
            timestamp: None,
            send_ack_time_gap_ms: None,
            entries: Vec::new(),
        };
        let encoded = encode_vec(&payload).expect("encode");
        assert_eq!(payload.size(), 3);

        let decoded: AckVectorPayload = decode(&encoded).expect("decode");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn ack_vector_roundtrip_mixed_entries() {
        let payload = AckVectorPayload {
            base_seq_num: 500,
            timestamp: Some(0x00_AABBCC),
            send_ack_time_gap_ms: Some(100),
            entries: vec![
                AckVectorEntry::RunLength {
                    received: true,
                    length: 63,
                },
                AckVectorEntry::StateMap { bitmap: 0b0111_1111 },
                AckVectorEntry::RunLength {
                    received: false,
                    length: 1,
                },
                AckVectorEntry::RunLength {
                    received: true,
                    length: 5,
                },
            ],
        };
        let encoded = encode_vec(&payload).expect("encode");
        let decoded: AckVectorPayload = decode(&encoded).expect("decode");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn ack_vector_insufficient_bytes() {
        let bytes = [0x00, 0x00]; // only 2 bytes, need 3
        let result: DecodeResult<AckVectorPayload> = decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn ack_vector_insufficient_entries() {
        // Claims 10 entries but provides none
        let bytes = [
            0x00, 0x00, // base_seq_num
            10,   // codedAckVecSize=10, TimeStampPresent=0
        ];
        let result: DecodeResult<AckVectorPayload> = decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn ack_vector_insufficient_timestamp_block() {
        // TimeStampPresent=1 but not enough bytes for the timestamp block
        let bytes = [
            0x00, 0x00, // base_seq_num
            0x80, // codedAckVecSize=0, TimeStampPresent=1
            0x05, // only 1 byte of the 4-byte timestamp block
        ];
        let result: DecodeResult<AckVectorPayload> = decode(&bytes);
        assert!(result.is_err());
    }
}
