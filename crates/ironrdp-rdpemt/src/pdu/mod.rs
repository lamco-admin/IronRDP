//! RDPEMT tunnel PDU definitions per MS-RDPEMT Section 2.2.
//!
//! Three PDU types share a common `TunnelHeader`:
//!
//! - [`TunnelCreateRequest`]: client → server tunnel binding (Section 2.2.2.1)
//! - [`TunnelCreateResponse`]: server → client confirmation (Section 2.2.2.2)
//! - [`TunnelData`]: bidirectional data transport (Section 2.2.2.3)
//!
//! The top-level [`TunnelPdu`] enum dispatches decoding based on the Action
//! nibble in byte 0 of the tunnel header.

pub mod create_request;
pub mod create_response;
pub mod data;
pub mod header;
pub mod multitransport_request;
pub mod multitransport_response;
pub mod subheader;

pub use create_request::{SECURITY_COOKIE_LEN, TunnelCreateRequest};
pub use create_response::TunnelCreateResponse;
pub use data::TunnelData;
pub use header::{TunnelAction, TunnelHeader};
use ironrdp_core::{Decode, DecodeResult, ReadCursor, UnexpectedMessageTypeErr as _};
pub use multitransport_request::{MultitransportRequest, RequestedProtocol};
pub use multitransport_response::MultitransportResponse;
pub use subheader::{SubHeaderType, TunnelSubHeader};

/// Discriminated union of all RDPEMT PDU types.
///
/// Decoded by reading the Action nibble from byte 0 and dispatching
/// to the appropriate variant's decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TunnelPdu {
    /// Client → server tunnel creation request.
    CreateRequest(TunnelCreateRequest),
    /// Server → client tunnel creation response.
    CreateResponse(TunnelCreateResponse),
    /// Bidirectional higher-layer data.
    Data(TunnelData),
}

impl Decode<'_> for TunnelPdu {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        // Peek at byte 0 to determine the action without consuming it,
        // since the individual PDU decoders expect the full wire bytes
        // starting from the header.
        ironrdp_core::ensure_size!(in: src, size: 1);
        let byte0 = src.remaining()[0];
        let action_raw = byte0 & 0x0F;

        let action = TunnelAction::from_u8(action_raw)
            .ok_or_else(|| ironrdp_core::DecodeError::unexpected_message_type("TunnelPdu", action_raw))?;

        match action {
            TunnelAction::CreateRequest => TunnelCreateRequest::decode(src).map(TunnelPdu::CreateRequest),
            TunnelAction::CreateResponse => TunnelCreateResponse::decode(src).map(TunnelPdu::CreateResponse),
            TunnelAction::Data => TunnelData::decode(src).map(TunnelPdu::Data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_create_request() {
        let wire: &[u8] = &[
            0x00, 0x18, 0x00, 0x04, // Header
            0x01, 0x00, 0x00, 0x00, // RequestID = 1
            0x00, 0x00, 0x00, 0x00, // Reserved
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Cookie (16 bytes)
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];

        let pdu: TunnelPdu = ironrdp_core::decode(wire).expect("decode");
        match pdu {
            TunnelPdu::CreateRequest(req) => {
                assert_eq!(req.request_id, 1);
            }
            other => panic!("expected CreateRequest, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_create_response() {
        let wire: &[u8] = &[
            0x01, 0x04, 0x00, 0x04, // Header
            0x00, 0x00, 0x00, 0x00, // HrResponse = S_OK
        ];

        let pdu: TunnelPdu = ironrdp_core::decode(wire).expect("decode");
        match pdu {
            TunnelPdu::CreateResponse(resp) => {
                assert!(resp.is_success());
            }
            other => panic!("expected CreateResponse, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_data() {
        let wire: &[u8] = &[
            0x02, 0x03, 0x00, 0x04, // Header: Data, payload=3, header=4
            0xAA, 0xBB, 0xCC, // HigherLayerData
        ];

        let pdu: TunnelPdu = ironrdp_core::decode(wire).expect("decode");
        match pdu {
            TunnelPdu::Data(data) => {
                assert_eq!(data.higher_layer_data, [0xAA, 0xBB, 0xCC]);
                assert!(data.sub_headers.is_empty());
            }
            other => panic!("expected Data, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_unknown_action() {
        let wire: &[u8] = &[0x0F, 0x00, 0x00, 0x04];
        let result: DecodeResult<TunnelPdu> = ironrdp_core::decode(wire);
        assert!(result.is_err());
    }
}
