//! RDP_TUNNEL_HEADER: common header for all RDPEMT PDUs.
//!
//! MS-RDPEMT Section 2.2.1.1.
//!
//! Wire layout (4+ bytes):
//! ```text
//! Byte 0:   [Action:4 bits (low nibble)] [Flags:4 bits (high nibble)]
//! Byte 1-2: PayloadLength (u16 LE)
//! Byte 3:   HeaderLength (u8)
//! Byte 4+:  SubHeaders (variable, present if HeaderLength > 4)
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use ironrdp_core::{
    Decode, DecodeResult, Encode, EncodeResult, InvalidFieldErr as _, ReadCursor, UnexpectedMessageTypeErr as _,
    WriteCursor,
};

use crate::pdu::subheader::TunnelSubHeader;

/// Tunnel PDU action type (low 4 bits of byte 0).
///
/// MS-RDPEMT Section 2.2.1.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TunnelAction {
    /// Client → server tunnel creation request.
    CreateRequest = 0x0,
    /// Server → client tunnel creation response.
    CreateResponse = 0x1,
    /// Bidirectional higher-layer data transfer.
    Data = 0x2,
}

impl TunnelAction {
    /// Attempt to parse an action value from the low nibble of byte 0.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x0 => Some(Self::CreateRequest),
            0x1 => Some(Self::CreateResponse),
            0x2 => Some(Self::Data),
            _ => None,
        }
    }

    /// Wire representation of this action (low nibble of byte 0).
    #[expect(clippy::as_conversions, reason = "repr(u8) enum to u8 is safe")]
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

/// Parsed RDP_TUNNEL_HEADER.
///
/// The header is at least 4 bytes. When `header_length > 4`, the remaining
/// bytes between position 4 and `header_length` contain SubHeader structures
/// used for auto-detect embedding (MS-RDPBCGR Section 2.2.14).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TunnelHeader {
    /// PDU type discriminator (low nibble of byte 0).
    pub action: TunnelAction,
    /// Bytes of payload following the header (does NOT include header size).
    pub payload_length: u16,
    /// Total header size in bytes, including any SubHeaders. Minimum 4.
    pub header_length: u8,
    /// Parsed sub-headers. Empty for CreateRequest/CreateResponse.
    pub sub_headers: Vec<TunnelSubHeader>,
}

impl TunnelHeader {
    /// Minimum header size: 1 (action+flags) + 2 (payload_length) + 1 (header_length).
    pub(crate) const MIN_SIZE: usize = 4;

    const NAME: &'static str = "RDP_TUNNEL_HEADER";
}

impl Encode for TunnelHeader {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_size!(in: dst, size: self.size());

        // Byte 0: (flags << 4) | action, flags are always 0
        dst.write_u8(self.action.to_u8());
        dst.write_u16(self.payload_length);
        dst.write_u8(self.header_length);

        for sub in &self.sub_headers {
            sub.encode(dst)?;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::MIN_SIZE + self.sub_headers.iter().map(TunnelSubHeader::size).sum::<usize>()
    }
}

impl Decode<'_> for TunnelHeader {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_size!(in: src, size: Self::MIN_SIZE);

        let byte0 = src.read_u8();
        let action_raw = byte0 & 0x0F;
        let flags = byte0 >> 4;

        // MS-RDPEMT Section 2.2.1.1: Flags "MUST be set to zero"
        if flags != 0 {
            return Err(ironrdp_core::DecodeError::invalid_field(
                Self::NAME,
                "Flags",
                "flags must be zero",
            ));
        }

        let action = TunnelAction::from_u8(action_raw)
            .ok_or_else(|| ironrdp_core::DecodeError::unexpected_message_type(Self::NAME, action_raw))?;

        let payload_length = src.read_u16();
        let header_length = src.read_u8();

        if header_length < 4 {
            return Err(ironrdp_core::DecodeError::invalid_field(
                Self::NAME,
                "HeaderLength",
                "header length must be at least 4",
            ));
        }

        // Parse sub-headers from the remaining header bytes
        let sub_header_bytes = usize::from(header_length) - Self::MIN_SIZE;
        let mut sub_headers = Vec::new();

        if sub_header_bytes > 0 {
            ironrdp_core::ensure_size!(in: src, size: sub_header_bytes);
            let sub_data = src.read_slice(sub_header_bytes);
            let mut sub_cursor = ReadCursor::new(sub_data);

            while !sub_cursor.is_empty() {
                let sub = TunnelSubHeader::decode(&mut sub_cursor)?;
                sub_headers.push(sub);
            }
        }

        Ok(Self {
            action,
            payload_length,
            header_length,
            sub_headers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_minimal_header() {
        let header = TunnelHeader {
            action: TunnelAction::CreateRequest,
            payload_length: 24,
            header_length: 4,
            sub_headers: Vec::new(),
        };

        let encoded = ironrdp_core::encode_vec(&header).expect("encode");
        assert_eq!(encoded, [0x00, 0x18, 0x00, 0x04]);

        let decoded: TunnelHeader = ironrdp_core::decode(&encoded).expect("decode");
        assert_eq!(decoded, header);
    }

    #[test]
    fn round_trip_create_response_header() {
        let header = TunnelHeader {
            action: TunnelAction::CreateResponse,
            payload_length: 4,
            header_length: 4,
            sub_headers: Vec::new(),
        };

        let encoded = ironrdp_core::encode_vec(&header).expect("encode");
        assert_eq!(encoded, [0x01, 0x04, 0x00, 0x04]);

        let decoded: TunnelHeader = ironrdp_core::decode(&encoded).expect("decode");
        assert_eq!(decoded, header);
    }

    #[test]
    fn round_trip_data_header() {
        let header = TunnelHeader {
            action: TunnelAction::Data,
            payload_length: 100,
            header_length: 4,
            sub_headers: Vec::new(),
        };

        let encoded = ironrdp_core::encode_vec(&header).expect("encode");
        assert_eq!(encoded, [0x02, 0x64, 0x00, 0x04]);

        let decoded: TunnelHeader = ironrdp_core::decode(&encoded).expect("decode");
        assert_eq!(decoded, header);
    }

    #[test]
    fn decode_rejects_nonzero_flags() {
        // Byte 0 = 0x10 means Action=0, Flags=1
        let wire = [0x10, 0x18, 0x00, 0x04];
        let result: DecodeResult<TunnelHeader> = ironrdp_core::decode(&wire);
        assert!(result.is_err());
    }

    #[test]
    fn decode_rejects_unknown_action() {
        // Action = 0x0F
        let wire = [0x0F, 0x04, 0x00, 0x04];
        let result: DecodeResult<TunnelHeader> = ironrdp_core::decode(&wire);
        assert!(result.is_err());
    }

    #[test]
    fn decode_rejects_header_too_small() {
        // HeaderLength = 3 (below minimum of 4)
        let wire = [0x00, 0x18, 0x00, 0x03];
        let result: DecodeResult<TunnelHeader> = ironrdp_core::decode(&wire);
        assert!(result.is_err());
    }

    #[test]
    fn decode_rejects_truncated_input() {
        let wire = [0x00, 0x18]; // only 2 bytes
        let result: DecodeResult<TunnelHeader> = ironrdp_core::decode(&wire);
        assert!(result.is_err());
    }

    #[test]
    fn round_trip_header_with_subheader() {
        let header = TunnelHeader {
            action: TunnelAction::Data,
            payload_length: 50,
            // 4 (base) + 4 (one subheader: len=4, type=1, data=[0xAA, 0xBB])
            header_length: 8,
            sub_headers: vec![TunnelSubHeader {
                sub_header_type: crate::pdu::subheader::SubHeaderType::AutoDetectResponse,
                data: vec![0xAA, 0xBB],
            }],
        };

        let encoded = ironrdp_core::encode_vec(&header).expect("encode");
        // byte 0: 0x02 (Data), payload_length: 50=0x0032, header_length: 8
        // subheader: length=4, type=0x01, data=0xAA 0xBB
        assert_eq!(encoded, [0x02, 0x32, 0x00, 0x08, 0x04, 0x01, 0xAA, 0xBB]);

        let decoded: TunnelHeader = ironrdp_core::decode(&encoded).expect("decode");
        assert_eq!(decoded, header);
    }
}
