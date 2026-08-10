//! Client Initiate Multitransport Response PDU payload.
//!
//! MS-RDPBCGR Section 2.2.15.2.
//!
//! This is the **inner payload** after the TPKT + X224 + MCS + security
//! header have been stripped. The client sends this on the MCS message
//! channel after the UDP tunnel is established (or has failed).
//!
//! Wire layout (8 bytes):
//! ```text
//! Bytes 0-3: RequestID (u32 LE)
//! Bytes 4-7: HrResponse (u32 LE)
//! ```

use ironrdp_core::{Decode, DecodeResult, Encode, EncodeResult, ReadCursor, WriteCursor};

/// Total payload size: RequestID(4) + HrResponse(4).
const WIRE_SIZE: usize = 4 + 4;

/// Client Initiate Multitransport Response PDU payload.
///
/// Sent by the client on the MCS message channel after the UDP
/// transport is established or after a failure to connect.
///
/// The `request_id` must match the corresponding Initiate
/// Multitransport Request PDU from the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultitransportResponse {
    /// Request ID matching the server's Initiate Multitransport Request.
    pub request_id: u32,
    /// Result code indicating success or failure.
    pub hr_response: u32,
}

impl MultitransportResponse {
    const NAME: &'static str = "MultitransportResponse";

    /// `S_OK`: multitransport connection successfully established.
    ///
    /// Per MS-RDPBCGR, this MUST only be sent to servers that advertised
    /// `SOFTSYNC_TCP_TO_UDP (0x200)` in the GCC MultiTransportChannelData.
    pub const S_OK: u32 = 0x0000_0000;

    /// `E_ABORT`: client was unable to establish the multitransport connection.
    pub const E_ABORT: u32 = 0x8000_4004;

    /// Create a success response for the given request ID.
    pub fn success(request_id: u32) -> Self {
        Self {
            request_id,
            hr_response: Self::S_OK,
        }
    }

    /// Create a failure response for the given request ID.
    pub fn abort(request_id: u32) -> Self {
        Self {
            request_id,
            hr_response: Self::E_ABORT,
        }
    }

    /// Whether this response indicates success.
    pub fn is_success(&self) -> bool {
        self.hr_response == Self::S_OK
    }
}

impl Encode for MultitransportResponse {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_size!(in: dst, size: self.size());

        dst.write_u32(self.request_id);
        dst.write_u32(self.hr_response);

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        WIRE_SIZE
    }
}

impl Decode<'_> for MultitransportResponse {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_size!(in: src, size: WIRE_SIZE);

        let request_id = src.read_u32();
        let hr_response = src.read_u32();

        Ok(Self {
            request_id,
            hr_response,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S_OK response for request ID 42.
    const WIRE_SUCCESS: &[u8] = &[
        0x2A, 0x00, 0x00, 0x00, // RequestID = 42
        0x00, 0x00, 0x00, 0x00, // HrResponse = S_OK
    ];

    /// E_ABORT response for request ID 7.
    const WIRE_ABORT: &[u8] = &[
        0x07, 0x00, 0x00, 0x00, // RequestID = 7
        0x04, 0x40, 0x00, 0x80, // HrResponse = E_ABORT (0x80004004 LE)
    ];

    #[test]
    fn decode_success() {
        let pdu: MultitransportResponse = ironrdp_core::decode(WIRE_SUCCESS).expect("decode");
        assert_eq!(pdu.request_id, 42);
        assert_eq!(pdu.hr_response, MultitransportResponse::S_OK);
        assert!(pdu.is_success());
    }

    #[test]
    fn decode_abort() {
        let pdu: MultitransportResponse = ironrdp_core::decode(WIRE_ABORT).expect("decode");
        assert_eq!(pdu.request_id, 7);
        assert_eq!(pdu.hr_response, MultitransportResponse::E_ABORT);
        assert!(!pdu.is_success());
    }

    #[test]
    fn encode_success_matches_wire() {
        let pdu = MultitransportResponse::success(42);
        let encoded = ironrdp_core::encode_vec(&pdu).expect("encode");
        assert_eq!(encoded.as_slice(), WIRE_SUCCESS);
    }

    #[test]
    fn encode_abort_matches_wire() {
        let pdu = MultitransportResponse::abort(7);
        let encoded = ironrdp_core::encode_vec(&pdu).expect("encode");
        assert_eq!(encoded.as_slice(), WIRE_ABORT);
    }

    #[test]
    fn round_trip_success() {
        let original = MultitransportResponse::success(0xDEAD_BEEF);
        let encoded = ironrdp_core::encode_vec(&original).expect("encode");
        let decoded: MultitransportResponse = ironrdp_core::decode(&encoded).expect("decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn round_trip_abort() {
        let original = MultitransportResponse::abort(123);
        let encoded = ironrdp_core::encode_vec(&original).expect("encode");
        let decoded: MultitransportResponse = ironrdp_core::decode(&encoded).expect("decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn wire_size_is_8() {
        let pdu = MultitransportResponse::success(0);
        assert_eq!(pdu.size(), 8);
    }

    #[test]
    fn constructor_convenience() {
        let s = MultitransportResponse::success(10);
        assert_eq!(s.request_id, 10);
        assert!(s.is_success());

        let a = MultitransportResponse::abort(20);
        assert_eq!(a.request_id, 20);
        assert!(!a.is_success());
    }

    #[test]
    fn unknown_hr_response_round_trips() {
        let pdu = MultitransportResponse {
            request_id: 1,
            hr_response: 0x1234_5678,
        };
        let encoded = ironrdp_core::encode_vec(&pdu).expect("encode");
        let decoded: MultitransportResponse = ironrdp_core::decode(&encoded).expect("decode");
        assert_eq!(decoded, pdu);
        assert!(!decoded.is_success());
    }
}
