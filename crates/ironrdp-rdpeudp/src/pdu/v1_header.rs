//! RDPUDP_FEC_HEADER: the mandatory header for every v1 datagram.
//!
//! MS-RDPEUDP Section 2.2.2.1.
//! Wire layout: `snSourceAck(4)` + `uReceiveWindowSize(2)` + `uFlags(2)` = 8 bytes, little-endian.

use ironrdp_core::{Decode, DecodeResult, Encode, EncodeResult, ReadCursor, WriteCursor};

use super::v1_flags::V1Flags;

/// Common header present on every v1 datagram.
///
/// MS-RDPEUDP Section 2.2.2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FecHeader {
    /// Highest sequence number the remote endpoint has received.
    /// Set to `0xFFFFFFFF` in the initial SYN.
    pub sn_source_ack: u32,

    /// Size of the receiver's buffer (in packets).
    pub receive_window_size: u16,

    /// Bitmap of `V1Flags` indicating optional payloads and features.
    pub flags: V1Flags,
}

impl FecHeader {
    const FIXED_PART_SIZE: usize = 4 + 2 + 2; // snSourceAck + uReceiveWindowSize + uFlags

    const NAME: &'static str = "RDPUDP_FEC_HEADER";
}

impl Encode for FecHeader {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_fixed_part_size!(in: dst);

        dst.write_u32(self.sn_source_ack);
        dst.write_u16(self.receive_window_size);
        dst.write_u16(self.flags.bits());

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE
    }
}

impl Decode<'_> for FecHeader {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_fixed_part_size!(in: src);

        let sn_source_ack = src.read_u32();
        let receive_window_size = src.read_u16();
        let flags_raw = src.read_u16();

        let flags = V1Flags::from_bits_truncate(flags_raw);

        Ok(Self {
            sn_source_ack,
            receive_window_size,
            flags,
        })
    }
}

#[cfg(test)]
mod tests {
    use ironrdp_core::{decode, encode_vec};

    use super::*;

    /// SYN datagram header per MS-RDPEUDP Section 3.1.5.1.1:
    /// snSourceAck = 0xFFFFFFFF, flags = SYN | SYNEX
    const SYN_HEADER_BYTES: [u8; 8] = [
        0xFF, 0xFF, 0xFF, 0xFF, // snSourceAck = -1 (u32::MAX)
        0x40, 0x00, // uReceiveWindowSize = 64
        0x01, 0x10, // uFlags = SYN(0x0001) | SYNEX(0x1000) = 0x1001
    ];

    fn syn_header() -> FecHeader {
        FecHeader {
            sn_source_ack: 0xFFFF_FFFF,
            receive_window_size: 64,
            flags: V1Flags::SYN | V1Flags::SYNEX,
        }
    }

    #[test]
    fn encode_syn_header() {
        let encoded = encode_vec(&syn_header()).expect("encode should succeed");
        assert_eq!(encoded.as_slice(), &SYN_HEADER_BYTES);
    }

    #[test]
    fn decode_syn_header() {
        let decoded: FecHeader = decode(&SYN_HEADER_BYTES).expect("decode should succeed");
        assert_eq!(decoded, syn_header());
    }

    #[test]
    fn roundtrip() {
        let original = FecHeader {
            sn_source_ack: 0x0000_1234,
            receive_window_size: 128,
            flags: V1Flags::ACK | V1Flags::CN | V1Flags::ACK_OF_ACKS,
        };
        let encoded = encode_vec(&original).expect("encode");
        let decoded: FecHeader = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn size_matches_encoding() {
        let header = syn_header();
        let encoded = encode_vec(&header).expect("encode");
        assert_eq!(header.size(), encoded.len());
    }

    #[test]
    fn decode_insufficient_bytes() {
        let short = [0xFF, 0xFF, 0xFF]; // only 3 bytes, need 8
        let result: DecodeResult<FecHeader> = decode(&short);
        assert!(result.is_err());
    }
}
