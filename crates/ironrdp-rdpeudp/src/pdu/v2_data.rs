//! V2 data payloads.
//!
//! `DataHeader Payload` (MS-RDPEUDP2 Section 2.2.1.2.5) and
//! `DataBody Payload` (MS-RDPEUDP2 Section 2.2.1.2.7).
//!
//! Both are present when the DATA flag (0x004) is set.
//! DataHeader always precedes DataBody in the packet layout.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use ironrdp_core::{Decode, DecodeResult, Encode, EncodeResult, ReadCursor, WriteCursor};

// ── DataHeader Payload ──

/// Header portion of a data payload, carrying the data sequence number.
///
/// MS-RDPEUDP2 Section 2.2.1.2.5.
/// Wire layout: `DataSeqNum(2)` = 2 bytes.
///
/// `DataSeqNum` is the lower 16 bits of the coded sequence number.
/// This value changes on retransmit: a retransmitted packet gets
/// a new DataSeqNum but preserves its ChannelSeqNum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataHeader {
    /// Lower 16 bits of the data sequence number.
    pub data_seq_num: u16,
}

impl DataHeader {
    const FIXED_PART_SIZE: usize = 2;
    const NAME: &'static str = "RDP-UDP2 DataHeader Payload";
}

impl Encode for DataHeader {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_fixed_part_size!(in: dst);
        dst.write_u16(self.data_seq_num);
        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE
    }
}

impl Decode<'_> for DataHeader {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_fixed_part_size!(in: src);
        let data_seq_num = src.read_u16();
        Ok(Self { data_seq_num })
    }
}

// ── DataBody Payload ──

/// Application data payload, containing the channel sequence number
/// and the actual RDP data from higher layers.
///
/// MS-RDPEUDP2 Section 2.2.1.2.7.
/// Wire layout: `ChannelSeqNum(2)` + `Data(variable)`.
///
/// `ChannelSeqNum` is preserved across retransmits: the reliability
/// controller uses it to match a retransmitted packet to the original.
/// This is the sequence number that higher layers care about for
/// ordering and reassembly. See MS-RDPEUDP2 Section 3.1.1.2.4.
///
/// The `Data` field contains the RDP protocol data from upper layers
/// and extends to the end of the packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataBody {
    /// Lower 16 bits of the channel sequence number.
    /// Stable across retransmits.
    pub channel_seq_num: u16,

    /// Application data from higher RDP layers.
    pub data: Vec<u8>,
}

impl DataBody {
    const FIXED_PART_SIZE: usize = 2; // ChannelSeqNum
    const NAME: &'static str = "RDP-UDP2 DataBody Payload";
}

impl Encode for DataBody {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_size!(in: dst, size: self.size());
        dst.write_u16(self.channel_seq_num);
        dst.write_slice(&self.data);
        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE + self.data.len()
    }
}

impl<'de> Decode<'de> for DataBody {
    fn decode(src: &mut ReadCursor<'de>) -> DecodeResult<Self> {
        ironrdp_core::ensure_fixed_part_size!(in: src);
        let channel_seq_num = src.read_u16();
        let data = src.read_remaining().to_vec();
        Ok(Self { channel_seq_num, data })
    }
}

#[cfg(test)]
mod tests {
    use ironrdp_core::{decode, encode_vec};

    use super::*;

    // ── DataHeader tests ──

    #[test]
    fn data_header_encode() {
        let header = DataHeader { data_seq_num: 0x1234 };
        let encoded = encode_vec(&header).expect("encode");
        assert_eq!(encoded.as_slice(), &[0x34, 0x12]); // little-endian
    }

    #[test]
    fn data_header_decode() {
        let bytes = [0x34, 0x12];
        let decoded: DataHeader = decode(&bytes).expect("decode");
        assert_eq!(decoded.data_seq_num, 0x1234);
    }

    #[test]
    fn data_header_roundtrip() {
        let original = DataHeader { data_seq_num: 0xFFFF };
        let encoded = encode_vec(&original).expect("encode");
        let decoded: DataHeader = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn data_header_size() {
        let header = DataHeader { data_seq_num: 0 };
        assert_eq!(header.size(), 2);
    }

    #[test]
    fn data_header_insufficient_bytes() {
        let bytes = [0x34]; // only 1 byte, need 2
        let result: DecodeResult<DataHeader> = decode(&bytes);
        assert!(result.is_err());
    }

    // ── DataBody tests ──

    #[test]
    fn data_body_with_payload() {
        let body = DataBody {
            channel_seq_num: 0x0042,
            data: vec![0x01, 0x02, 0x03, 0x04],
        };
        let encoded = encode_vec(&body).expect("encode");
        assert_eq!(
            encoded.as_slice(),
            &[
                0x42, 0x00, // channel_seq_num = 0x0042
                0x01, 0x02, 0x03, 0x04, // data
            ]
        );
        assert_eq!(body.size(), 6);
    }

    #[test]
    fn data_body_empty_data() {
        let body = DataBody {
            channel_seq_num: 0,
            data: Vec::new(),
        };
        let encoded = encode_vec(&body).expect("encode");
        assert_eq!(encoded.as_slice(), &[0x00, 0x00]);
        assert_eq!(body.size(), 2);
    }

    #[test]
    fn data_body_roundtrip() {
        let original = DataBody {
            channel_seq_num: 0xABCD,
            data: vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE],
        };
        let encoded = encode_vec(&original).expect("encode");
        let decoded: DataBody = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn data_body_large_payload() {
        // Simulate a near-MTU payload
        let original = DataBody {
            channel_seq_num: 1,
            data: vec![0xAA; 1200],
        };
        let encoded = encode_vec(&original).expect("encode");
        let decoded: DataBody = decode(&encoded).expect("decode");
        assert_eq!(original.data.len(), decoded.data.len());
        assert_eq!(original, decoded);
    }

    #[test]
    fn data_body_insufficient_bytes() {
        let bytes = [0x42]; // only 1 byte, need 2 for ChannelSeqNum
        let result: DecodeResult<DataBody> = decode(&bytes);
        assert!(result.is_err());
    }

    // ── Combined DataHeader + DataBody test ──

    #[test]
    fn data_header_then_body_sequential() {
        // Simulate reading a DATA payload: DataHeader then DataBody
        let header = DataHeader { data_seq_num: 100 };
        let body = DataBody {
            channel_seq_num: 50,
            data: vec![0x11, 0x22, 0x33],
        };

        let mut buf = vec![0u8; header.size() + body.size()];
        let mut cursor = WriteCursor::new(&mut buf);
        header.encode(&mut cursor).expect("encode header");
        body.encode(&mut cursor).expect("encode body");

        let mut read_cursor = ReadCursor::new(&buf);
        let decoded_header = DataHeader::decode(&mut read_cursor).expect("decode header");
        let decoded_body = DataBody::decode(&mut read_cursor).expect("decode body");

        assert_eq!(decoded_header, header);
        assert_eq!(decoded_body, body);
    }
}
