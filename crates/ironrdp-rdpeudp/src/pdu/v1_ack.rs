//! V1 ACK-related structures.
//!
//! `RDPUDP_ACK_VECTOR_HEADER` (MS-RDPEUDP Section 2.2.2.7) and
//! `RDPUDP_ACK_OF_ACKVECTOR_HEADER` (MS-RDPEUDP Section 2.2.2.6).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use ironrdp_core::{Decode, DecodeResult, Encode, EncodeResult, ReadCursor, WriteCursor};

// -- ACK Vector Element --

#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
/// State of a received packet in the ACK vector.
///
/// MS-RDPEUDP Section 2.2.2.7.1: each element is 1 byte,
/// high bit = state (1 = received, 0 = not received),
/// low 7 bits = count of consecutive packets in that state (0..=127).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V1AckVectorElement {
    /// `true` = packets in this run were received.
    pub received: bool,

    /// Number of consecutive packets in this state (0..=127).
    pub length: u8,
}

impl V1AckVectorElement {
    /// Maximum run length in a single element.
    pub const MAX_LENGTH: u8 = 0x7F;

    /// Encode to a single byte: MSB = state, low 7 bits = length.
    fn to_byte(self) -> u8 {
        let state_bit = if self.received { 0x80 } else { 0x00 };
        state_bit | (self.length & 0x7F)
    }

    /// Decode from a single byte.
    fn from_byte(byte: u8) -> Self {
        Self {
            received: (byte & 0x80) != 0,
            length: byte & 0x7F,
        }
    }
}

// -- RDPUDP_ACK_VECTOR_HEADER --

#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
/// ACK vector describing the state of the receiver's packet queue.
///
/// MS-RDPEUDP Section 2.2.2.7.
/// Wire layout: `uAckVectorSize(2)` + `AckVectorElement[](variable)`.
///
/// The elements are run-length encoded: each element has a 1-bit state
/// and a 7-bit run length, describing consecutive packets that were
/// either all received or all not received.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V1AckVectorHeader {
    /// RLE-encoded state of packets in the receiver queue.
    pub elements: Vec<V1AckVectorElement>,
}

impl V1AckVectorHeader {
    const FIXED_PART_SIZE: usize = 2; // uAckVectorSize
    const NAME: &'static str = "RDPUDP_ACK_VECTOR_HEADER";
}

impl Encode for V1AckVectorHeader {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        let count: u16 = ironrdp_core::cast_length!("RDPUDP_ACK_VECTOR_HEADER", "uAckVectorSize", self.elements.len())?;

        ironrdp_core::ensure_size!(in: dst, size: self.size());

        dst.write_u16(count);
        for element in &self.elements {
            dst.write_u8(element.to_byte());
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE + self.elements.len()
    }
}

impl Decode<'_> for V1AckVectorHeader {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_fixed_part_size!(in: src);

        let count = src.read_u16();
        let count_usize = usize::from(count);

        ironrdp_core::ensure_size!(in: src, size: count_usize);

        let mut elements = Vec::with_capacity(count_usize);
        for _ in 0..count_usize {
            elements.push(V1AckVectorElement::from_byte(src.read_u8()));
        }

        Ok(Self { elements })
    }
}

// -- RDPUDP_ACK_OF_ACKVECTOR_HEADER --

#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
/// Resets the starting position of ACK vector encoding at the receiver.
///
/// MS-RDPEUDP Section 2.2.2.6.
/// Wire layout: `snResetSeqNum(4)` = 4 bytes.
///
/// Sent after approximately every 20 packets. The receiver generates
/// ACK vectors only for sequence numbers greater than `reset_seq_num`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V1AckOfAcksHeader {
    /// Sequence number to reset the ACK vector base to.
    /// The sender populates this with the greatest cumulative ACK
    /// it has received and processed.
    pub reset_seq_num: u32,
}

impl V1AckOfAcksHeader {
    const FIXED_PART_SIZE: usize = 4;
    const NAME: &'static str = "RDPUDP_ACK_OF_ACKVECTOR_HEADER";
}

impl Encode for V1AckOfAcksHeader {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_fixed_part_size!(in: dst);
        dst.write_u32(self.reset_seq_num);
        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE
    }
}

impl Decode<'_> for V1AckOfAcksHeader {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_fixed_part_size!(in: src);
        let reset_seq_num = src.read_u32();
        Ok(Self { reset_seq_num })
    }
}

// -- RDPUDP_CORRELATION_ID_PAYLOAD --

#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
/// Correlation ID for diagnostic binding between TCP and UDP connections.
///
/// MS-RDPEUDP Section 2.2.2.8.
/// Wire layout: `uCorrelationId(16)` = 16 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorrelationIdPayload {
    /// A 16-byte identifier that correlates the UDP connection
    /// with its parent TCP connection for diagnostic purposes.
    pub correlation_id: [u8; 16],
}

impl CorrelationIdPayload {
    const FIXED_PART_SIZE: usize = 16;
    const NAME: &'static str = "RDPUDP_CORRELATION_ID_PAYLOAD";
}

impl Encode for CorrelationIdPayload {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_fixed_part_size!(in: dst);
        dst.write_slice(&self.correlation_id);
        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE
    }
}

impl Decode<'_> for CorrelationIdPayload {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_fixed_part_size!(in: src);
        let correlation_id = src.read_array::<16>();
        Ok(Self { correlation_id })
    }
}

#[cfg(test)]
mod tests {
    use ironrdp_core::{decode, encode_vec};

    use super::*;

    // -- V1AckVectorElement tests --

    #[test]
    fn ack_vector_element_received() {
        let elem = V1AckVectorElement {
            received: true,
            length: 42,
        };
        let byte = elem.to_byte();
        assert_eq!(byte, 0x80 | 42);
        let decoded = V1AckVectorElement::from_byte(byte);
        assert_eq!(decoded, elem);
    }

    #[test]
    fn ack_vector_element_not_received() {
        let elem = V1AckVectorElement {
            received: false,
            length: 3,
        };
        let byte = elem.to_byte();
        assert_eq!(byte, 3);
        let decoded = V1AckVectorElement::from_byte(byte);
        assert_eq!(decoded, elem);
    }

    #[test]
    fn ack_vector_element_max_length() {
        let elem = V1AckVectorElement {
            received: true,
            length: V1AckVectorElement::MAX_LENGTH,
        };
        let byte = elem.to_byte();
        assert_eq!(byte, 0xFF);
        let decoded = V1AckVectorElement::from_byte(byte);
        assert_eq!(decoded, elem);
    }

    // -- V1AckVectorHeader tests --

    #[test]
    fn encode_ack_vector_empty() {
        let header = V1AckVectorHeader { elements: Vec::new() };
        let encoded = encode_vec(&header).expect("encode");
        assert_eq!(encoded.as_slice(), &[0x00, 0x00]); // count = 0
        assert_eq!(header.size(), 2);
    }

    #[test]
    fn encode_ack_vector_with_elements() {
        let header = V1AckVectorHeader {
            elements: vec![
                V1AckVectorElement {
                    received: true,
                    length: 10,
                },
                V1AckVectorElement {
                    received: false,
                    length: 2,
                },
                V1AckVectorElement {
                    received: true,
                    length: 5,
                },
            ],
        };
        let encoded = encode_vec(&header).expect("encode");
        assert_eq!(
            encoded.as_slice(),
            &[
                0x03,
                0x00,      // count = 3
                0x80 | 10, // received, length 10
                2,         // not received, length 2
                0x80 | 5,  // received, length 5
            ]
        );
        assert_eq!(header.size(), 5);
    }

    #[test]
    fn ack_vector_roundtrip() {
        let original = V1AckVectorHeader {
            elements: vec![
                V1AckVectorElement {
                    received: true,
                    length: 127,
                },
                V1AckVectorElement {
                    received: false,
                    length: 1,
                },
            ],
        };
        let encoded = encode_vec(&original).expect("encode");
        let decoded: V1AckVectorHeader = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn ack_vector_insufficient_bytes_for_count() {
        let bytes = [0x01]; // only 1 byte, need 2 for count
        let result: DecodeResult<V1AckVectorHeader> = decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn ack_vector_insufficient_bytes_for_elements() {
        let bytes = [0x05, 0x00, 0x80]; // claims 5 elements, only has 1
        let result: DecodeResult<V1AckVectorHeader> = decode(&bytes);
        assert!(result.is_err());
    }

    // -- V1AckOfAcksHeader tests --

    #[test]
    fn ack_of_acks_roundtrip() {
        let original = V1AckOfAcksHeader {
            reset_seq_num: 0xDEAD_BEEF,
        };
        let encoded = encode_vec(&original).expect("encode");
        assert_eq!(
            encoded.as_slice(),
            &[0xEF, 0xBE, 0xAD, 0xDE] // little-endian
        );
        let decoded: V1AckOfAcksHeader = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn ack_of_acks_size() {
        let header = V1AckOfAcksHeader { reset_seq_num: 0 };
        assert_eq!(header.size(), 4);
    }

    // -- CorrelationIdPayload tests --

    #[test]
    fn correlation_id_roundtrip() {
        let original = CorrelationIdPayload {
            correlation_id: [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
            ],
        };
        let encoded = encode_vec(&original).expect("encode");
        assert_eq!(encoded.len(), 16);
        let decoded: CorrelationIdPayload = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn correlation_id_insufficient_bytes() {
        let bytes = [0x01, 0x02, 0x03]; // only 3 bytes, need 16
        let result: DecodeResult<CorrelationIdPayload> = decode(&bytes);
        assert!(result.is_err());
    }
}
