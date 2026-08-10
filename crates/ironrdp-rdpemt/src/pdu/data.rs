//! RDP_TUNNEL_DATA: bidirectional higher-layer data transport.
//!
//! MS-RDPEMT Section 2.2.2.3.
//!
//! Wire layout (variable size):
//! ```text
//! Bytes 0-3:             TunnelHeader (Action=0x2, HeaderLength >= 0x04)
//! Bytes 4..HeaderLength: SubHeaders (if HeaderLength > 4)
//! Bytes HeaderLength..:  HigherLayerData (PayloadLength bytes)
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use ironrdp_core::{
    Decode, DecodeResult, Encode, EncodeResult, ReadCursor, UnexpectedMessageTypeErr as _, WriteCursor,
};

use crate::pdu::header::{TunnelAction, TunnelHeader};
use crate::pdu::subheader::TunnelSubHeader;

#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
/// Tunnel Data PDU.
///
/// Wraps higher-layer data (DVC traffic) with the tunnel header.
/// Optional SubHeaders may be present for auto-detect piggy-backing
/// (bandwidth measurement per MS-RDPBCGR Section 2.2.14).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TunnelData {
    /// Optional sub-headers for auto-detect embedding.
    pub sub_headers: Vec<TunnelSubHeader>,
    /// Opaque higher-layer data (DVC payload).
    pub higher_layer_data: Vec<u8>,
}

impl TunnelData {
    const NAME: &'static str = "RDP_TUNNEL_DATA";

    fn header_length(&self) -> u8 {
        let sub_size: usize = self.sub_headers.iter().map(TunnelSubHeader::wire_size).sum();
        // 4 (base header) + sub-headers
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "header length fits in u8 per spec"
        )]
        let len = (TunnelHeader::MIN_SIZE + sub_size) as u8;
        len
    }
}

impl Encode for TunnelData {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_size!(in: dst, size: self.size());

        let header = TunnelHeader {
            action: TunnelAction::Data,
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "payload capped at u16::MAX by callers"
            )]
            payload_length: self.higher_layer_data.len() as u16,
            header_length: self.header_length(),
            sub_headers: self.sub_headers.clone(),
        };
        header.encode(dst)?;

        dst.write_slice(&self.higher_layer_data);

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        usize::from(self.header_length()) + self.higher_layer_data.len()
    }
}

impl Decode<'_> for TunnelData {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        let header = TunnelHeader::decode(src)?;

        if header.action != TunnelAction::Data {
            return Err(ironrdp_core::DecodeError::unexpected_message_type(
                Self::NAME,
                header.action.to_u8(),
            ));
        }

        let payload_len = usize::from(header.payload_length);
        ironrdp_core::ensure_size!(in: src, size: payload_len);
        let higher_layer_data = src.read_slice(payload_len).to_vec();

        Ok(Self {
            sub_headers: header.sub_headers,
            higher_layer_data,
        })
    }
}
