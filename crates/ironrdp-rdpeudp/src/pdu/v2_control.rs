//! V2 control payloads: OverheadSize, DelayAckInfo, AckOfAcks.
//!
//! These are small fixed-size payloads used for flow control and
//! protocol tuning during data transfer.

use ironrdp_core::{Decode, DecodeResult, Encode, EncodeResult, ReadCursor, WriteCursor};

// ── OverheadSize Payload ──

/// Average extra header bytes from the RDP-UDP2 layer to the raw UDP layer.
///
/// MS-RDPEUDP2 Section 2.2.1.2.2.
/// Present when the OVERHEADSIZE flag (0x040) is set.
/// Sent by the Receiver to help the Sender's congestion control
/// accurately estimate available bandwidth.
///
/// Wire layout: `OverheadSize(1)` = 1 byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverheadSizePayload {
    /// Average header overhead in bytes.
    pub overhead_size: u8,
}

impl OverheadSizePayload {
    const FIXED_PART_SIZE: usize = 1;
    const NAME: &'static str = "RDP-UDP2 OverheadSize Payload";
}

impl Encode for OverheadSizePayload {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_fixed_part_size!(in: dst);
        dst.write_u8(self.overhead_size);
        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE
    }
}

impl Decode<'_> for OverheadSizePayload {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_fixed_part_size!(in: src);
        let overhead_size = src.read_u8();
        Ok(Self { overhead_size })
    }
}

// ── DelayAckInfo Payload ──

/// Parameters controlling the Receiver's ACK batching behavior.
///
/// MS-RDPEUDP2 Section 2.2.1.2.3.
/// Present when the DELAYACKINFO flag (0x100) is set.
/// Sent by the Sender to configure how the Receiver batches ACKs.
///
/// Wire layout (3 bytes):
/// - `MaxDelayedAcks(4 bits) : reserved(4 bits)` = 1 byte.
/// - `DelayedAckTimeoutInMs(16 bits)` = 2 bytes.
///
/// Default values (assumed after initialization):
/// - `MaxDelayedAcks = 8`
/// - `DelayedAckTimeoutInMs = RTT / 2`
///
/// The Sender can adjust these at any time during the connection.
/// If the Receiver delays beyond these parameters, the packet is
/// considered lost and must be retransmitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelayAckInfoPayload {
    /// Maximum number of ACKs the Receiver may batch before
    /// sending a standalone acknowledgment. Range 0..=15.
    pub max_delayed_acks: u8,

    /// Timeout in milliseconds. If the Receiver has unacknowledged
    /// packets older than this, it must send an ACK immediately.
    pub delayed_ack_timeout_ms: u16,
}

impl DelayAckInfoPayload {
    const FIXED_PART_SIZE: usize = 3;
    const NAME: &'static str = "RDP-UDP2 DelayAckInfo Payload";

    /// Default max delayed ACKs per MS-RDPEUDP2 Section 3.1.5.2.
    pub const DEFAULT_MAX_DELAYED_ACKS: u8 = 8;
}

impl Encode for DelayAckInfoPayload {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_fixed_part_size!(in: dst);

        // High nibble = MaxDelayedAcks, low nibble = reserved (0)
        let packed = (self.max_delayed_acks & 0x0F) << 4;
        dst.write_u8(packed);
        dst.write_u16(self.delayed_ack_timeout_ms);

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE
    }
}

impl Decode<'_> for DelayAckInfoPayload {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_fixed_part_size!(in: src);

        let packed = src.read_u8();
        let max_delayed_acks = (packed >> 4) & 0x0F;
        let delayed_ack_timeout_ms = src.read_u16();

        Ok(Self {
            max_delayed_acks,
            delayed_ack_timeout_ms,
        })
    }
}

// ── AckOfAcks Payload ──

/// Tells the Receiver the lowest sequence number the Sender still cares about.
///
/// MS-RDPEUDP2 Section 2.2.1.2.4.
/// Present when the AOA flag (0x010) is set.
/// Sent by the Sender to allow the Receiver to shrink its window buffer.
///
/// Wire layout: `AckOfAcksSeqNum(2)` = 2 bytes.
///
/// The Sender sends this after detecting packet loss to inform the Receiver
/// that packets with lower sequence numbers need not be tracked anymore.
/// The Sender should stop sending this once it receives an ACK or ACK vector
/// confirming a higher sequence number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AckOfAcksPayload {
    /// Lower 16 bits of the AckOfAcks sequence number.
    pub ack_of_acks_seq_num: u16,
}

impl AckOfAcksPayload {
    const FIXED_PART_SIZE: usize = 2;
    const NAME: &'static str = "RDP-UDP2 AckOfAcks Payload";
}

impl Encode for AckOfAcksPayload {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_fixed_part_size!(in: dst);
        dst.write_u16(self.ack_of_acks_seq_num);
        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE
    }
}

impl Decode<'_> for AckOfAcksPayload {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_fixed_part_size!(in: src);
        let ack_of_acks_seq_num = src.read_u16();
        Ok(Self { ack_of_acks_seq_num })
    }
}

#[cfg(test)]
mod tests {
    use ironrdp_core::{decode, encode_vec};

    use super::*;

    // ── OverheadSize tests ──

    #[test]
    fn overhead_size_roundtrip() {
        let original = OverheadSizePayload { overhead_size: 42 };
        let encoded = encode_vec(&original).expect("encode");
        assert_eq!(encoded.as_slice(), &[42]);
        assert_eq!(original.size(), 1);

        let decoded: OverheadSizePayload = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn overhead_size_zero() {
        let original = OverheadSizePayload { overhead_size: 0 };
        let encoded = encode_vec(&original).expect("encode");
        let decoded: OverheadSizePayload = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn overhead_size_max() {
        let original = OverheadSizePayload { overhead_size: 0xFF };
        let encoded = encode_vec(&original).expect("encode");
        let decoded: OverheadSizePayload = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn overhead_size_insufficient_bytes() {
        let bytes: [u8; 0] = [];
        let result: DecodeResult<OverheadSizePayload> = decode(&bytes);
        assert!(result.is_err());
    }

    // ── DelayAckInfo tests ──

    #[test]
    fn delay_ack_info_encode() {
        let payload = DelayAckInfoPayload {
            max_delayed_acks: 8,
            delayed_ack_timeout_ms: 150,
        };
        let encoded = encode_vec(&payload).expect("encode");
        assert_eq!(
            encoded.as_slice(),
            &[
                0x80, // max_delayed_acks=8 << 4 = 0x80
                0x96, 0x00, // delayed_ack_timeout_ms = 150 LE
            ]
        );
        assert_eq!(payload.size(), 3);
    }

    #[test]
    fn delay_ack_info_decode() {
        let bytes = [0x80, 0x96, 0x00];
        let decoded: DelayAckInfoPayload = decode(&bytes).expect("decode");
        assert_eq!(decoded.max_delayed_acks, 8);
        assert_eq!(decoded.delayed_ack_timeout_ms, 150);
    }

    #[test]
    fn delay_ack_info_roundtrip() {
        let original = DelayAckInfoPayload {
            max_delayed_acks: 15,
            delayed_ack_timeout_ms: 1000,
        };
        let encoded = encode_vec(&original).expect("encode");
        let decoded: DelayAckInfoPayload = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn delay_ack_info_default_values() {
        let payload = DelayAckInfoPayload {
            max_delayed_acks: DelayAckInfoPayload::DEFAULT_MAX_DELAYED_ACKS,
            delayed_ack_timeout_ms: 50,
        };
        assert_eq!(payload.max_delayed_acks, 8);
    }

    #[test]
    fn delay_ack_info_insufficient_bytes() {
        let bytes = [0x80, 0x96]; // only 2 bytes, need 3
        let result: DecodeResult<DelayAckInfoPayload> = decode(&bytes);
        assert!(result.is_err());
    }

    // ── AckOfAcks tests ──

    #[test]
    fn ack_of_acks_roundtrip() {
        let original = AckOfAcksPayload {
            ack_of_acks_seq_num: 0x1234,
        };
        let encoded = encode_vec(&original).expect("encode");
        assert_eq!(encoded.as_slice(), &[0x34, 0x12]); // little-endian
        assert_eq!(original.size(), 2);

        let decoded: AckOfAcksPayload = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn ack_of_acks_zero() {
        let original = AckOfAcksPayload { ack_of_acks_seq_num: 0 };
        let encoded = encode_vec(&original).expect("encode");
        let decoded: AckOfAcksPayload = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn ack_of_acks_max() {
        let original = AckOfAcksPayload {
            ack_of_acks_seq_num: 0xFFFF,
        };
        let encoded = encode_vec(&original).expect("encode");
        let decoded: AckOfAcksPayload = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn ack_of_acks_insufficient_bytes() {
        let bytes = [0x34]; // only 1 byte, need 2
        let result: DecodeResult<AckOfAcksPayload> = decode(&bytes);
        assert!(result.is_err());
    }
}
