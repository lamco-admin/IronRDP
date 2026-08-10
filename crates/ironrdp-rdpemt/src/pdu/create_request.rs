//! RDP_TUNNEL_CREATEREQUEST: client → server tunnel binding.
//!
//! MS-RDPEMT Section 2.2.2.1.
//!
//! Wire layout (28 bytes total):
//! ```text
//! Bytes 0-3:   TunnelHeader (Action=0x0, PayloadLength=0x0018, HeaderLength=0x04)
//! Bytes 4-7:   RequestID (u32 LE)
//! Bytes 8-11:  Reserved (u32 LE, MUST be 0)
//! Bytes 12-27: SecurityCookie (16 bytes)
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use ironrdp_core::{
    Decode, DecodeResult, Encode, EncodeResult, InvalidFieldErr as _, ReadCursor, UnexpectedMessageTypeErr as _,
    WriteCursor,
};

use crate::pdu::header::{TunnelAction, TunnelHeader};

/// Length of the security cookie (MS-RDPEMT Section 2.2.2.1).
pub const SECURITY_COOKIE_LEN: usize = 16;

/// Payload size: RequestID(4) + Reserved(4) + SecurityCookie(16) = 24.
const PAYLOAD_SIZE: usize = 4 + 4 + SECURITY_COOKIE_LEN;

/// Tunnel Create Request PDU.
///
/// Sent by the client after the TLS handshake to bind this UDP tunnel
/// to a main RDP session. The `request_id` and `security_cookie` must
/// match the Initiate Multitransport Request PDU sent by the server
/// over the main TCP connection (MS-RDPBCGR Section 2.2.15.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TunnelCreateRequest {
    /// Request ID from the Initiate Multitransport Request PDU.
    pub request_id: u32,
    /// 16-byte security cookie from the Initiate Multitransport Request PDU.
    pub security_cookie: [u8; SECURITY_COOKIE_LEN],
}

impl TunnelCreateRequest {
    const NAME: &'static str = "RDP_TUNNEL_CREATEREQUEST";

    /// Total wire size: 4 (header) + 24 (payload) = 28 bytes.
    const WIRE_SIZE: usize = TunnelHeader::MIN_SIZE + PAYLOAD_SIZE;
}

impl Encode for TunnelCreateRequest {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_size!(in: dst, size: self.size());

        let header = TunnelHeader {
            action: TunnelAction::CreateRequest,
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "PAYLOAD_SIZE is 24; fits in u16"
            )]
            payload_length: PAYLOAD_SIZE as u16,
            header_length: 4,
            sub_headers: Vec::new(),
        };
        header.encode(dst)?;

        dst.write_u32(self.request_id);
        dst.write_u32(0); // Reserved
        dst.write_slice(&self.security_cookie);

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::WIRE_SIZE
    }
}

impl Decode<'_> for TunnelCreateRequest {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        let header = TunnelHeader::decode(src)?;

        if header.action != TunnelAction::CreateRequest {
            return Err(ironrdp_core::DecodeError::unexpected_message_type(
                Self::NAME,
                header.action.to_u8(),
            ));
        }

        if header.header_length != 4 {
            return Err(ironrdp_core::DecodeError::invalid_field(
                Self::NAME,
                "HeaderLength",
                "must be 4 for CreateRequest",
            ));
        }

        ironrdp_core::ensure_size!(in: src, size: PAYLOAD_SIZE);

        let request_id = src.read_u32();
        let _reserved = src.read_u32();
        let security_cookie: [u8; SECURITY_COOKIE_LEN] = src.read_array();

        Ok(Self {
            request_id,
            security_cookie,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Protocol example from MS-RDPEMT Section 4.1.
    const SPEC_EXAMPLE: &[u8] = &[
        0x00, 0x18, 0x00, 0x04, // Header: Action=0, PayloadLen=24, HeaderLen=4
        0x07, 0x00, 0x00, 0x00, // RequestID = 7
        0x00, 0x00, 0x00, 0x00, // Reserved = 0
        0xe2, 0xf0, 0xd1, 0x08, 0x56, 0x7f, 0xb4, 0x3a, // SecurityCookie (16 bytes)
        0xdc, 0xf4, 0xb3, 0xdc, 0x16, 0x92, 0x1e, 0x3a,
    ];

    #[test]
    fn decode_spec_example() {
        let pdu: TunnelCreateRequest = ironrdp_core::decode(SPEC_EXAMPLE).expect("decode");
        assert_eq!(pdu.request_id, 7);
        assert_eq!(
            pdu.security_cookie,
            [
                0xe2, 0xf0, 0xd1, 0x08, 0x56, 0x7f, 0xb4, 0x3a, 0xdc, 0xf4, 0xb3, 0xdc, 0x16, 0x92, 0x1e, 0x3a
            ]
        );
    }

    #[test]
    fn encode_matches_spec_example() {
        let pdu = TunnelCreateRequest {
            request_id: 7,
            security_cookie: [
                0xe2, 0xf0, 0xd1, 0x08, 0x56, 0x7f, 0xb4, 0x3a, 0xdc, 0xf4, 0xb3, 0xdc, 0x16, 0x92, 0x1e, 0x3a,
            ],
        };

        let encoded = ironrdp_core::encode_vec(&pdu).expect("encode");
        assert_eq!(encoded.as_slice(), SPEC_EXAMPLE);
    }

    #[test]
    fn round_trip() {
        let original = TunnelCreateRequest {
            request_id: 0xDEAD_BEEF,
            security_cookie: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        };

        let encoded = ironrdp_core::encode_vec(&original).expect("encode");
        let decoded: TunnelCreateRequest = ironrdp_core::decode(&encoded).expect("decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn wire_size_is_28() {
        let pdu = TunnelCreateRequest {
            request_id: 0,
            security_cookie: [0; 16],
        };
        assert_eq!(pdu.size(), 28);
    }
}
