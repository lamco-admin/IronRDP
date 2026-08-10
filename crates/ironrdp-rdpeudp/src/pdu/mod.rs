//! RDPEUDP PDU definitions.
//!
//! V1 handshake format (MS-RDPEUDP Section 2.2) and
//! V2 data transfer format (MS-RDPEUDP2 Section 2.2).
//!
//! The v1 format is used for the three-way handshake (SYN/SYN+ACK/ACK)
//! even when both sides negotiate v2+ for data transfer. Once the
//! handshake completes, all subsequent packets use the v2 wire format
//! from MS-RDPEUDP2.
//!
//! Types are organized by protocol version:
//! - `v1_*` modules: MS-RDPEUDP handshake structures.
//! - `v2_*` modules: MS-RDPEUDP2 data transfer structures.
//! - `prefix`: PacketPrefixByte wire-level framing for v2 packets.

use ironrdp_core::{Decode, DecodeResult, Encode, EncodeResult, ReadCursor, WriteCursor};

// ── V1 Handshake modules ──

pub mod v1_ack;
pub mod v1_flags;
pub mod v1_header;
pub mod v1_syn;

// ── V2 Data Transfer modules ──

pub mod v2_ack;
pub mod v2_control;
pub mod v2_data;
pub mod v2_flags;
pub mod v2_header;

// ── Wire-level framing ──

pub mod prefix;

// ── V1 re-exports ──

// ── Prefix re-exports ──
pub use prefix::{PacketPrefixByte, PrefixError, decode_with_prefix, encode_with_prefix};
pub use v1_ack::{CorrelationIdPayload, V1AckOfAcksHeader, V1AckVectorElement, V1AckVectorHeader};
pub use v1_flags::V1Flags;
pub use v1_header::FecHeader;
pub use v1_syn::{MTU_MAX, MTU_MIN, SynDataExPayload, SynDataPayload, SynExFlags, UdpVersion};
// ── V2 re-exports ──
pub use v2_ack::{AckPayload, AckVectorEntry, AckVectorPayload};
pub use v2_control::{AckOfAcksPayload, DelayAckInfoPayload, OverheadSizePayload};
pub use v2_data::{DataBody, DataHeader};
pub use v2_flags::V2Flags;
pub use v2_header::{LOG_WINDOW_SIZE_MAX, V2Header};

// ════════════════════════════════════════════════════════════════════
// Composite V1 Datagram
// ════════════════════════════════════════════════════════════════════

/// V1 flags that don't gate any optional payload; preserved on encode.
///
/// DATA and FEC are payload-gating but have no corresponding fields
/// in V1Datagram (v1 data transfer is not supported).
const V1_STANDALONE_FLAGS: u16 = V1Flags::FIN.bits()
    | V1Flags::CN.bits()
    | V1Flags::CWR.bits()
    | V1Flags::SACK_OPTION.bits()
    | V1Flags::SYNLOSSY.bits()
    | V1Flags::ACKDELAYED.bits();

/// A complete v1 datagram (SYN, SYN+ACK, or ACK).
///
/// MS-RDPEUDP Section 2.2.
/// The FecHeader's flags field determines which optional payloads
/// are present. On encode, payload-gating flags are automatically
/// derived from which `Option` fields are populated; standalone flags
/// (FIN, CN, CWR, SACK_OPTION, SYNLOSSY, ACKDELAYED) are preserved
/// from the header.
///
/// Wire payload ordering (per MS-RDPEUDP Section 2.2.2):
/// 1. FecHeader (mandatory, 8 bytes)
/// 2. V1AckVectorHeader (if ACK flag)
/// 3. V1AckOfAcksHeader (if ACK_OF_ACKS flag)
/// 4. SynDataPayload (if SYN flag)
/// 5. CorrelationIdPayload (if CORRELATION_ID flag)
/// 6. SynDataExPayload (if SYNEX flag)
///
/// V1 data payloads (SOURCE_PAYLOAD / FEC_PAYLOAD) are not represented;
/// this crate always negotiates v2+ for data transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V1Datagram {
    /// Mandatory FecHeader. Standalone flags (FIN, CN, CWR, etc.)
    /// are preserved; payload-gating flags are recomputed on encode.
    pub header: FecHeader,

    /// ACK vector (run-length encoded receiver state).
    /// Gated by `V1Flags::ACK`.
    pub ack_vector: Option<V1AckVectorHeader>,

    /// AckOfAcks (resets ACK vector encoding base).
    /// Gated by `V1Flags::ACK_OF_ACKS`.
    pub ack_of_acks: Option<V1AckOfAcksHeader>,

    /// SYN data (ISN, MTU).
    /// Gated by `V1Flags::SYN`.
    pub syn_data: Option<SynDataPayload>,

    /// Correlation ID for TCP/UDP binding.
    /// Gated by `V1Flags::CORRELATION_ID`.
    pub correlation_id: Option<CorrelationIdPayload>,

    /// Extended SYN data (version negotiation).
    /// Gated by `V1Flags::SYNEX`.
    pub syn_data_ex: Option<SynDataExPayload>,
}

impl V1Datagram {
    const NAME: &'static str = "V1 Datagram";

    /// Compute the flags from populated fields, preserving standalone flags.
    fn compute_flags(&self) -> V1Flags {
        let mut flags = V1Flags::from_bits_truncate(self.header.flags.bits() & V1_STANDALONE_FLAGS);

        if self.ack_vector.is_some() {
            flags |= V1Flags::ACK;
        }
        if self.ack_of_acks.is_some() {
            flags |= V1Flags::ACK_OF_ACKS;
        }
        if self.syn_data.is_some() {
            flags |= V1Flags::SYN;
        }
        if self.correlation_id.is_some() {
            flags |= V1Flags::CORRELATION_ID;
        }
        if self.syn_data_ex.is_some() {
            flags |= V1Flags::SYNEX;
        }

        flags
    }
}

impl Encode for V1Datagram {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_size!(in: dst, size: self.size());

        // Write header with auto-computed flags
        let header = FecHeader {
            flags: self.compute_flags(),
            ..self.header
        };
        header.encode(dst)?;

        // Write payloads in spec-mandated order (MS-RDPEUDP Section 2.2.2)
        if let Some(ref ack_vector) = self.ack_vector {
            ack_vector.encode(dst)?;
        }
        if let Some(ref ack_of_acks) = self.ack_of_acks {
            ack_of_acks.encode(dst)?;
        }
        if let Some(ref syn_data) = self.syn_data {
            syn_data.encode(dst)?;
        }
        if let Some(ref correlation_id) = self.correlation_id {
            correlation_id.encode(dst)?;
        }
        if let Some(ref syn_data_ex) = self.syn_data_ex {
            syn_data_ex.encode(dst)?;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        let mut total = self.header.size();
        if let Some(ref av) = self.ack_vector {
            total += av.size();
        }
        if let Some(ref aoa) = self.ack_of_acks {
            total += aoa.size();
        }
        if let Some(ref sd) = self.syn_data {
            total += sd.size();
        }
        if let Some(ref cid) = self.correlation_id {
            total += cid.size();
        }
        if let Some(ref sdex) = self.syn_data_ex {
            total += sdex.size();
        }
        total
    }
}

impl Decode<'_> for V1Datagram {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        let header = FecHeader::decode(src)?;

        // V1 data payloads are not supported; reject if present since we
        // cannot skip them without knowing their wire size.
        if header.flags.contains(V1Flags::DATA) {
            return Err(ironrdp_core::invalid_field_err!(
                "V1 Datagram",
                "flags",
                "DATA flag is not supported in handshake datagrams"
            ));
        }
        if header.flags.contains(V1Flags::FEC) {
            return Err(ironrdp_core::invalid_field_err!(
                "V1 Datagram",
                "flags",
                "FEC flag is not supported in handshake datagrams"
            ));
        }

        // Decode payloads in spec-mandated order, gated by flags
        let ack_vector = if header.flags.contains(V1Flags::ACK) {
            Some(V1AckVectorHeader::decode(src)?)
        } else {
            None
        };

        let ack_of_acks = if header.flags.contains(V1Flags::ACK_OF_ACKS) {
            Some(V1AckOfAcksHeader::decode(src)?)
        } else {
            None
        };

        let syn_data = if header.flags.contains(V1Flags::SYN) {
            Some(SynDataPayload::decode(src)?)
        } else {
            None
        };

        let correlation_id = if header.flags.contains(V1Flags::CORRELATION_ID) {
            Some(CorrelationIdPayload::decode(src)?)
        } else {
            None
        };

        let syn_data_ex = if header.flags.contains(V1Flags::SYNEX) {
            Some(SynDataExPayload::decode(src)?)
        } else {
            None
        };

        Ok(Self {
            header,
            ack_vector,
            ack_of_acks,
            syn_data,
            correlation_id,
            syn_data_ex,
        })
    }
}

// ════════════════════════════════════════════════════════════════════
// Composite V2 Packet
// ════════════════════════════════════════════════════════════════════

/// V2 flags that don't gate any optional payload; preserved on encode.
const V2_STANDALONE_FLAGS: u16 = V2Flags::CN.bits() | V2Flags::CWR.bits() | V2Flags::DUMMY.bits();

/// A complete RDP-UDP2 packet (after prefix byte extraction).
///
/// MS-RDPEUDP2 Section 2.2.1.
/// The V2Header's flags field determines which optional payloads
/// are present. On encode, payload-gating flags are auto-derived;
/// standalone flags (CN, CWR, DUMMY) are preserved from the header.
///
/// Wire payload ordering (per MS-RDPEUDP2 Section 2.2.1):
/// 1. V2Header (mandatory, 2 bytes)
/// 2. AckPayload (if ACK flag)
/// 3. OverheadSizePayload (if OVERHEADSIZE flag)
/// 4. DelayAckInfoPayload (if DELAYACKINFO flag)
/// 5. AckOfAcksPayload (if AOA flag)
/// 6. DataHeader (if DATA flag)
/// 7. AckVectorPayload (if ACKVEC flag)
/// 8. DataBody (if DATA flag), always last, extending to the end of the packet
///
/// Invariants:
/// - `ack` and `ack_vector` are mutually exclusive.
/// - `data_header` and `data_body` must be both present or both absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2Packet {
    /// Mandatory header. Standalone flags (CN, CWR, DUMMY) are
    /// preserved; payload-gating flags are recomputed on encode.
    pub header: V2Header,

    /// ACK payload. Gated by `V2Flags::ACK`.
    /// Mutually exclusive with `ack_vector`.
    pub ack: Option<AckPayload>,

    /// OverheadSize payload. Gated by `V2Flags::OVERHEADSIZE`.
    pub overhead_size: Option<OverheadSizePayload>,

    /// DelayAckInfo payload. Gated by `V2Flags::DELAYACKINFO`.
    pub delay_ack_info: Option<DelayAckInfoPayload>,

    /// AckOfAcks payload. Gated by `V2Flags::AOA`.
    pub ack_of_acks: Option<AckOfAcksPayload>,

    /// DataHeader payload. Gated by `V2Flags::DATA`.
    /// Must be paired with `data_body`.
    pub data_header: Option<DataHeader>,

    /// AckVector payload. Gated by `V2Flags::ACKVEC`.
    /// Mutually exclusive with `ack`.
    pub ack_vector: Option<AckVectorPayload>,

    /// DataBody payload. Gated by `V2Flags::DATA`.
    /// Must be paired with `data_header`.
    /// Always last in wire layout (consumes remaining bytes on decode).
    pub data_body: Option<DataBody>,
}

impl V2Packet {
    const NAME: &'static str = "RDP-UDP2 Packet";

    /// Compute flags from populated fields, preserving standalone flags.
    fn compute_flags(&self) -> V2Flags {
        let mut flags = V2Flags::from_bits_truncate(self.header.flags.bits() & V2_STANDALONE_FLAGS);

        if self.ack.is_some() {
            flags |= V2Flags::ACK;
        }
        if self.overhead_size.is_some() {
            flags |= V2Flags::OVERHEADSIZE;
        }
        if self.delay_ack_info.is_some() {
            flags |= V2Flags::DELAYACKINFO;
        }
        if self.ack_of_acks.is_some() {
            flags |= V2Flags::AOA;
        }
        if self.data_header.is_some() {
            flags |= V2Flags::DATA;
        }
        if self.ack_vector.is_some() {
            flags |= V2Flags::ACKVEC;
        }

        flags
    }
}

impl Encode for V2Packet {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        // Validate invariants before writing anything
        if self.data_header.is_some() != self.data_body.is_some() {
            return Err(ironrdp_core::invalid_field_err!(
                "V2Packet",
                "DATA",
                "data_header and data_body must be both present or both absent"
            ));
        }
        if self.ack.is_some() && self.ack_vector.is_some() {
            return Err(ironrdp_core::invalid_field_err!(
                "V2Packet",
                "ACK/ACKVEC",
                "ACK and ACKVEC are mutually exclusive"
            ));
        }

        ironrdp_core::ensure_size!(in: dst, size: self.size());

        // Write header with auto-computed flags
        let header = V2Header {
            flags: self.compute_flags(),
            ..self.header
        };
        header.encode(dst)?;

        // Write payloads in spec-mandated order (MS-RDPEUDP2 Section 2.2.1)
        if let Some(ref ack) = self.ack {
            ack.encode(dst)?;
        }
        if let Some(ref overhead) = self.overhead_size {
            overhead.encode(dst)?;
        }
        if let Some(ref dai) = self.delay_ack_info {
            dai.encode(dst)?;
        }
        if let Some(ref aoa) = self.ack_of_acks {
            aoa.encode(dst)?;
        }
        if let Some(ref dh) = self.data_header {
            dh.encode(dst)?;
        }
        if let Some(ref av) = self.ack_vector {
            av.encode(dst)?;
        }
        if let Some(ref db) = self.data_body {
            db.encode(dst)?;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        let mut total = self.header.size();
        if let Some(ref a) = self.ack {
            total += a.size();
        }
        if let Some(ref os) = self.overhead_size {
            total += os.size();
        }
        if let Some(ref dai) = self.delay_ack_info {
            total += dai.size();
        }
        if let Some(ref aoa) = self.ack_of_acks {
            total += aoa.size();
        }
        if let Some(ref dh) = self.data_header {
            total += dh.size();
        }
        if let Some(ref av) = self.ack_vector {
            total += av.size();
        }
        if let Some(ref db) = self.data_body {
            total += db.size();
        }
        total
    }
}

impl Decode<'_> for V2Packet {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        // V2Header::decode validates ACK/ACKVEC mutual exclusion
        let header = V2Header::decode(src)?;

        let ack = if header.flags.contains(V2Flags::ACK) {
            Some(AckPayload::decode(src)?)
        } else {
            None
        };

        let overhead_size = if header.flags.contains(V2Flags::OVERHEADSIZE) {
            Some(OverheadSizePayload::decode(src)?)
        } else {
            None
        };

        let delay_ack_info = if header.flags.contains(V2Flags::DELAYACKINFO) {
            Some(DelayAckInfoPayload::decode(src)?)
        } else {
            None
        };

        let ack_of_acks = if header.flags.contains(V2Flags::AOA) {
            Some(AckOfAcksPayload::decode(src)?)
        } else {
            None
        };

        let data_header = if header.flags.contains(V2Flags::DATA) {
            Some(DataHeader::decode(src)?)
        } else {
            None
        };

        let ack_vector = if header.flags.contains(V2Flags::ACKVEC) {
            Some(AckVectorPayload::decode(src)?)
        } else {
            None
        };

        // DataBody is always last: it consumes all remaining bytes
        let data_body = if header.flags.contains(V2Flags::DATA) {
            Some(DataBody::decode(src)?)
        } else {
            None
        };

        Ok(Self {
            header,
            ack,
            overhead_size,
            delay_ack_info,
            ack_of_acks,
            data_header,
            ack_vector,
            data_body,
        })
    }
}

#[cfg(test)]
mod tests {
    use ironrdp_core::{decode, encode_vec};

    use super::*;

    // ════════════════════════════════════════════════════════════════
    // V1Datagram tests
    // ════════════════════════════════════════════════════════════════

    /// Client SYN: header + SYNDATA + SYNDATAEX.
    #[test]
    fn v1_syn_datagram_roundtrip() {
        let datagram = V1Datagram {
            header: FecHeader {
                sn_source_ack: 0xFFFF_FFFF,
                receive_window_size: 64,
                flags: V1Flags::SYN | V1Flags::SYNEX,
            },
            ack_vector: None,
            ack_of_acks: None,
            syn_data: Some(SynDataPayload {
                initial_sequence_number: 0x1234_5678,
                upstream_mtu: 1232,
                downstream_mtu: 1232,
            }),
            correlation_id: None,
            syn_data_ex: Some(SynDataExPayload {
                syn_ex_flags: SynExFlags::VERSION_INFO_VALID,
                udp_ver: UdpVersion::V2,
                cookie_hash: None,
            }),
        };

        // header(8) + syndata(8) + syndataex(4) = 20 bytes
        assert_eq!(datagram.size(), 20);

        let encoded = encode_vec(&datagram).expect("encode");
        assert_eq!(encoded.len(), 20);

        let decoded: V1Datagram = decode(&encoded).expect("decode");
        assert_eq!(decoded, datagram);
    }

    /// Server SYN+ACK: header + ack_vector + SYNDATA + SYNDATAEX.
    #[test]
    fn v1_syn_ack_datagram_roundtrip() {
        let datagram = V1Datagram {
            header: FecHeader {
                sn_source_ack: 100,
                receive_window_size: 64,
                flags: V1Flags::SYN | V1Flags::ACK | V1Flags::SYNEX,
            },
            ack_vector: Some(V1AckVectorHeader {
                elements: vec![V1AckVectorElement {
                    received: true,
                    length: 1,
                }],
            }),
            ack_of_acks: None,
            syn_data: Some(SynDataPayload {
                initial_sequence_number: 0xAABB_CCDD,
                upstream_mtu: 1200,
                downstream_mtu: 1200,
            }),
            correlation_id: None,
            syn_data_ex: Some(SynDataExPayload {
                syn_ex_flags: SynExFlags::VERSION_INFO_VALID,
                udp_ver: UdpVersion::V2,
                cookie_hash: None,
            }),
        };

        // header(8) + ack_vector(2+1=3) + syndata(8) + syndataex(4) = 23
        assert_eq!(datagram.size(), 23);

        let encoded = encode_vec(&datagram).expect("encode");
        let decoded: V1Datagram = decode(&encoded).expect("decode");
        assert_eq!(decoded, datagram);
    }

    /// Client final ACK: header + ack_vector + ack_of_acks.
    #[test]
    fn v1_ack_datagram_roundtrip() {
        let datagram = V1Datagram {
            header: FecHeader {
                sn_source_ack: 200,
                receive_window_size: 64,
                flags: V1Flags::ACK | V1Flags::ACK_OF_ACKS,
            },
            ack_vector: Some(V1AckVectorHeader {
                elements: vec![
                    V1AckVectorElement {
                        received: true,
                        length: 5,
                    },
                    V1AckVectorElement {
                        received: false,
                        length: 2,
                    },
                ],
            }),
            ack_of_acks: Some(V1AckOfAcksHeader { reset_seq_num: 150 }),
            syn_data: None,
            correlation_id: None,
            syn_data_ex: None,
        };

        // header(8) + ack_vector(2+2=4) + ack_of_acks(4) = 16
        assert_eq!(datagram.size(), 16);

        let encoded = encode_vec(&datagram).expect("encode");
        let decoded: V1Datagram = decode(&encoded).expect("decode");
        assert_eq!(decoded, datagram);
    }

    /// SYN with correlation ID.
    #[test]
    fn v1_syn_with_correlation_id_roundtrip() {
        let datagram = V1Datagram {
            header: FecHeader {
                sn_source_ack: 0xFFFF_FFFF,
                receive_window_size: 64,
                flags: V1Flags::SYN | V1Flags::SYNEX | V1Flags::CORRELATION_ID,
            },
            ack_vector: None,
            ack_of_acks: None,
            syn_data: Some(SynDataPayload {
                initial_sequence_number: 42,
                upstream_mtu: 1232,
                downstream_mtu: 1132,
            }),
            correlation_id: Some(CorrelationIdPayload {
                correlation_id: [
                    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
                ],
            }),
            syn_data_ex: Some(SynDataExPayload {
                syn_ex_flags: SynExFlags::VERSION_INFO_VALID,
                udp_ver: UdpVersion::V2,
                cookie_hash: None,
            }),
        };

        // header(8) + syndata(8) + correlation_id(16) + syndataex(4) = 36
        assert_eq!(datagram.size(), 36);

        let encoded = encode_vec(&datagram).expect("encode");
        let decoded: V1Datagram = decode(&encoded).expect("decode");
        assert_eq!(decoded, datagram);
    }

    /// V3 SYN with 32-byte cookie hash.
    #[test]
    fn v1_syn_v3_with_cookie_roundtrip() {
        let datagram = V1Datagram {
            header: FecHeader {
                sn_source_ack: 0xFFFF_FFFF,
                receive_window_size: 64,
                flags: V1Flags::SYN | V1Flags::SYNEX,
            },
            ack_vector: None,
            ack_of_acks: None,
            syn_data: Some(SynDataPayload {
                initial_sequence_number: 0xDEAD_BEEF,
                upstream_mtu: 1232,
                downstream_mtu: 1232,
            }),
            correlation_id: None,
            syn_data_ex: Some(SynDataExPayload {
                syn_ex_flags: SynExFlags::VERSION_INFO_VALID,
                udp_ver: UdpVersion::V3,
                cookie_hash: Some([0xAA; 32]),
            }),
        };

        // header(8) + syndata(8) + syndataex(4+32=36) = 52
        assert_eq!(datagram.size(), 52);

        let encoded = encode_vec(&datagram).expect("encode");
        let decoded: V1Datagram = decode(&encoded).expect("decode");
        assert_eq!(decoded, datagram);
    }

    /// Verify flags are auto-computed from populated fields.
    #[test]
    fn v1_flags_auto_computed_on_encode() {
        let datagram = V1Datagram {
            header: FecHeader {
                sn_source_ack: 0xFFFF_FFFF,
                receive_window_size: 64,
                // Caller only set SYN, but ack_vector is also populated
                flags: V1Flags::SYN,
            },
            ack_vector: Some(V1AckVectorHeader { elements: vec![] }),
            ack_of_acks: None,
            syn_data: Some(SynDataPayload {
                initial_sequence_number: 1,
                upstream_mtu: 1232,
                downstream_mtu: 1232,
            }),
            correlation_id: None,
            // syn_data_ex is None, so SYNEX should NOT be in flags
            syn_data_ex: None,
        };

        let encoded = encode_vec(&datagram).expect("encode");
        let decoded: V1Datagram = decode(&encoded).expect("decode");

        // ACK should be auto-added (ack_vector is Some)
        assert!(decoded.header.flags.contains(V1Flags::ACK));
        // SYN should be present (syn_data is Some)
        assert!(decoded.header.flags.contains(V1Flags::SYN));
        // SYNEX should NOT be set (syn_data_ex is None)
        assert!(!decoded.header.flags.contains(V1Flags::SYNEX));
    }

    /// Standalone flags (CN, CWR, ACKDELAYED) are preserved on encode.
    #[test]
    fn v1_standalone_flags_preserved() {
        let datagram = V1Datagram {
            header: FecHeader {
                sn_source_ack: 100,
                receive_window_size: 64,
                flags: V1Flags::ACK | V1Flags::CN | V1Flags::ACKDELAYED,
            },
            ack_vector: Some(V1AckVectorHeader {
                elements: vec![V1AckVectorElement {
                    received: true,
                    length: 10,
                }],
            }),
            ack_of_acks: None,
            syn_data: None,
            correlation_id: None,
            syn_data_ex: None,
        };

        let encoded = encode_vec(&datagram).expect("encode");
        let decoded: V1Datagram = decode(&encoded).expect("decode");

        assert!(decoded.header.flags.contains(V1Flags::CN));
        assert!(decoded.header.flags.contains(V1Flags::ACKDELAYED));
        assert!(decoded.header.flags.contains(V1Flags::ACK));
    }

    /// Reject datagrams with DATA flag (not supported in handshake).
    #[test]
    fn v1_decode_rejects_data_flag() {
        let datagram = V1Datagram {
            header: FecHeader {
                sn_source_ack: 0,
                receive_window_size: 64,
                flags: V1Flags::empty(),
            },
            ack_vector: None,
            ack_of_acks: None,
            syn_data: None,
            correlation_id: None,
            syn_data_ex: None,
        };
        let mut encoded = encode_vec(&datagram).expect("encode");

        // Manually set DATA flag in the wire bytes
        // Flags are at offset 6..8 in FecHeader (after snSourceAck(4) + windowSize(2))
        let flags_raw = u16::from_le_bytes([encoded[6], encoded[7]]);
        let modified = flags_raw | V1Flags::DATA.bits();
        encoded[6] = (modified & 0xFF) as u8;
        encoded[7] = ((modified >> 8) & 0xFF) as u8;

        let result: DecodeResult<V1Datagram> = decode(&encoded);
        assert!(result.is_err());
    }

    /// Minimal datagram: just a header with no payloads.
    #[test]
    fn v1_empty_datagram_roundtrip() {
        let datagram = V1Datagram {
            header: FecHeader {
                sn_source_ack: 0,
                receive_window_size: 32,
                flags: V1Flags::empty(),
            },
            ack_vector: None,
            ack_of_acks: None,
            syn_data: None,
            correlation_id: None,
            syn_data_ex: None,
        };

        assert_eq!(datagram.size(), 8); // header only

        let encoded = encode_vec(&datagram).expect("encode");
        let decoded: V1Datagram = decode(&encoded).expect("decode");
        assert_eq!(decoded, datagram);
    }

    /// Insufficient bytes for header.
    #[test]
    fn v1_insufficient_bytes() {
        let bytes = [0x00, 0x00, 0x00]; // need 8 for header
        let result: DecodeResult<V1Datagram> = decode(&bytes);
        assert!(result.is_err());
    }

    // ════════════════════════════════════════════════════════════════
    // V2Packet tests
    // ════════════════════════════════════════════════════════════════

    /// ACK-only packet.
    #[test]
    fn v2_ack_only_roundtrip() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::ACK,
                log_window_size: 6,
            },
            ack: Some(AckPayload {
                seq_num: 0x0042,
                received_ts: 0x00_123456,
                send_ack_time_gap: 5,
                num_delayed_acks: 0,
                delay_ack_time_scale: 0,
                delay_ack_time_additions: Vec::new(),
            }),
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: None,
            ack_vector: None,
            data_body: None,
        };

        // header(2) + ack(7) = 9
        assert_eq!(packet.size(), 9);

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");
        assert_eq!(decoded, packet);
    }

    /// DATA-only packet.
    #[test]
    fn v2_data_only_roundtrip() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::DATA,
                log_window_size: 8,
            },
            ack: None,
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: Some(DataHeader { data_seq_num: 0x0100 }),
            ack_vector: None,
            data_body: Some(DataBody {
                channel_seq_num: 0x0001,
                data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            }),
        };

        // header(2) + data_header(2) + data_body(2+4=6) = 10
        assert_eq!(packet.size(), 10);

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");
        assert_eq!(decoded, packet);
    }

    /// ACK + DATA combined.
    #[test]
    fn v2_ack_with_data_roundtrip() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::ACK | V2Flags::DATA,
                log_window_size: 6,
            },
            ack: Some(AckPayload {
                seq_num: 50,
                received_ts: 0x00_AABBCC,
                send_ack_time_gap: 10,
                num_delayed_acks: 2,
                delay_ack_time_scale: 3,
                delay_ack_time_additions: vec![15, 20],
            }),
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: Some(DataHeader { data_seq_num: 0x1234 }),
            ack_vector: None,
            data_body: Some(DataBody {
                channel_seq_num: 100,
                data: vec![0x01, 0x02, 0x03],
            }),
        };

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");
        assert_eq!(decoded, packet);
    }

    /// ACKVEC packet (mutually exclusive with ACK).
    #[test]
    fn v2_ackvec_roundtrip() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::ACKVEC,
                log_window_size: 10,
            },
            ack: None,
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: None,
            ack_vector: Some(AckVectorPayload {
                base_seq_num: 0x0100,
                timestamp: Some(0x00_AABBCC),
                send_ack_time_gap_ms: Some(25),
                entries: vec![
                    AckVectorEntry::RunLength {
                        received: true,
                        length: 20,
                    },
                    AckVectorEntry::StateMap { bitmap: 0b0110_1010 },
                ],
            }),
            data_body: None,
        };

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");
        assert_eq!(decoded, packet);
    }

    /// All control payloads at once: OverheadSize + DelayAckInfo + AOA.
    #[test]
    fn v2_all_control_payloads_roundtrip() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::ACK | V2Flags::OVERHEADSIZE | V2Flags::DELAYACKINFO | V2Flags::AOA,
                log_window_size: 6,
            },
            ack: Some(AckPayload {
                seq_num: 10,
                received_ts: 500_000,
                send_ack_time_gap: 3,
                num_delayed_acks: 0,
                delay_ack_time_scale: 0,
                delay_ack_time_additions: Vec::new(),
            }),
            overhead_size: Some(OverheadSizePayload { overhead_size: 28 }),
            delay_ack_info: Some(DelayAckInfoPayload {
                max_delayed_acks: 8,
                delayed_ack_timeout_ms: 150,
            }),
            ack_of_acks: Some(AckOfAcksPayload { ack_of_acks_seq_num: 5 }),
            data_header: None,
            ack_vector: None,
            data_body: None,
        };

        // header(2) + ack(7) + overhead(1) + delay_ack_info(3) + aoa(2) = 15
        assert_eq!(packet.size(), 15);

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");
        assert_eq!(decoded, packet);
    }

    /// Full packet with ACK + OverheadSize + DelayAckInfo + AOA + DATA.
    #[test]
    fn v2_full_packet_roundtrip() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::ACK | V2Flags::OVERHEADSIZE | V2Flags::DELAYACKINFO | V2Flags::AOA | V2Flags::DATA,
                log_window_size: 8,
            },
            ack: Some(AckPayload {
                seq_num: 0x1000,
                received_ts: 0x00_FFFFFF,
                send_ack_time_gap: 50,
                num_delayed_acks: 3,
                delay_ack_time_scale: 2,
                delay_ack_time_additions: vec![10, 20, 30],
            }),
            overhead_size: Some(OverheadSizePayload { overhead_size: 40 }),
            delay_ack_info: Some(DelayAckInfoPayload {
                max_delayed_acks: 15,
                delayed_ack_timeout_ms: 500,
            }),
            ack_of_acks: Some(AckOfAcksPayload {
                ack_of_acks_seq_num: 0x0FF0,
            }),
            data_header: Some(DataHeader { data_seq_num: 0x2000 }),
            ack_vector: None,
            data_body: Some(DataBody {
                channel_seq_num: 0x1000,
                data: vec![0x55; 100],
            }),
        };

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");
        assert_eq!(decoded, packet);
    }

    /// Verify flags are auto-computed from populated fields.
    #[test]
    fn v2_flags_auto_computed_on_encode() {
        let packet = V2Packet {
            header: V2Header {
                // Caller only set DATA, but ack is also populated
                flags: V2Flags::DATA,
                log_window_size: 6,
            },
            ack: Some(AckPayload {
                seq_num: 1,
                received_ts: 0,
                send_ack_time_gap: 0,
                num_delayed_acks: 0,
                delay_ack_time_scale: 0,
                delay_ack_time_additions: Vec::new(),
            }),
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: Some(DataHeader { data_seq_num: 1 }),
            ack_vector: None,
            data_body: Some(DataBody {
                channel_seq_num: 1,
                data: vec![0x42],
            }),
        };

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");

        // ACK should be auto-added
        assert!(decoded.header.flags.contains(V2Flags::ACK));
        // DATA should be present
        assert!(decoded.header.flags.contains(V2Flags::DATA));
        // ACKVEC should NOT be set
        assert!(!decoded.header.flags.contains(V2Flags::ACKVEC));
    }

    /// Standalone flags (CN, CWR, DUMMY) are preserved on encode.
    #[test]
    fn v2_standalone_flags_preserved() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::ACK | V2Flags::CN | V2Flags::CWR,
                log_window_size: 6,
            },
            ack: Some(AckPayload {
                seq_num: 1,
                received_ts: 0,
                send_ack_time_gap: 0,
                num_delayed_acks: 0,
                delay_ack_time_scale: 0,
                delay_ack_time_additions: Vec::new(),
            }),
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: None,
            ack_vector: None,
            data_body: None,
        };

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");

        assert!(decoded.header.flags.contains(V2Flags::CN));
        assert!(decoded.header.flags.contains(V2Flags::CWR));
        assert!(decoded.header.flags.contains(V2Flags::ACK));
    }

    /// Encode rejects ACK + ACKVEC both set.
    #[test]
    fn v2_ack_and_ackvec_mutual_exclusion_encode() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::empty(),
                log_window_size: 6,
            },
            ack: Some(AckPayload {
                seq_num: 1,
                received_ts: 0,
                send_ack_time_gap: 0,
                num_delayed_acks: 0,
                delay_ack_time_scale: 0,
                delay_ack_time_additions: Vec::new(),
            }),
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: None,
            ack_vector: Some(AckVectorPayload {
                base_seq_num: 0,
                timestamp: None,
                send_ack_time_gap_ms: None,
                entries: Vec::new(),
            }),
            data_body: None,
        };

        let result = encode_vec(&packet);
        assert!(result.is_err());
    }

    /// Encode rejects data_header without data_body.
    #[test]
    fn v2_data_header_without_body_rejected() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::DATA,
                log_window_size: 6,
            },
            ack: None,
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: Some(DataHeader { data_seq_num: 1 }),
            ack_vector: None,
            data_body: None, // Missing!
        };

        let result = encode_vec(&packet);
        assert!(result.is_err());
    }

    /// Encode rejects data_body without data_header.
    #[test]
    fn v2_data_body_without_header_rejected() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::DATA,
                log_window_size: 6,
            },
            ack: None,
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: None, // Missing!
            ack_vector: None,
            data_body: Some(DataBody {
                channel_seq_num: 1,
                data: vec![0x01],
            }),
        };

        let result = encode_vec(&packet);
        assert!(result.is_err());
    }

    /// Empty packet: just header, no payloads.
    #[test]
    fn v2_empty_packet_roundtrip() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::empty(),
                log_window_size: 6,
            },
            ack: None,
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: None,
            ack_vector: None,
            data_body: None,
        };

        assert_eq!(packet.size(), 2); // header only

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");
        assert_eq!(decoded, packet);
    }

    /// Insufficient bytes for V2 header.
    #[test]
    fn v2_insufficient_bytes() {
        let bytes = [0x00]; // need 2 for header
        let result: DecodeResult<V2Packet> = decode(&bytes);
        assert!(result.is_err());
    }

    /// DataBody correctly consumes all remaining bytes.
    #[test]
    fn v2_data_body_consumes_remaining() {
        // Build a packet with ACK + DATA, then verify data_body gets all remaining
        let payload_data = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::ACK | V2Flags::DATA,
                log_window_size: 6,
            },
            ack: Some(AckPayload {
                seq_num: 1,
                received_ts: 100,
                send_ack_time_gap: 0,
                num_delayed_acks: 0,
                delay_ack_time_scale: 0,
                delay_ack_time_additions: Vec::new(),
            }),
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: Some(DataHeader { data_seq_num: 1 }),
            ack_vector: None,
            data_body: Some(DataBody {
                channel_seq_num: 1,
                data: payload_data.clone(),
            }),
        };

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");

        let body = decoded.data_body.expect("data_body should be present");
        assert_eq!(body.data, payload_data);
    }

    /// ACKVEC + DATA combined (no ACK).
    #[test]
    fn v2_ackvec_with_data_roundtrip() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::ACKVEC | V2Flags::DATA,
                log_window_size: 6,
            },
            ack: None,
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: Some(DataHeader { data_seq_num: 0x0042 }),
            ack_vector: Some(AckVectorPayload {
                base_seq_num: 0x0100,
                timestamp: None,
                send_ack_time_gap_ms: None,
                entries: vec![AckVectorEntry::RunLength {
                    received: true,
                    length: 10,
                }],
            }),
            data_body: Some(DataBody {
                channel_seq_num: 0x0042,
                data: vec![0xAA, 0xBB, 0xCC],
            }),
        };

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");
        assert_eq!(decoded, packet);
    }

    /// Dummy packet with DATA.
    #[test]
    fn v2_dummy_packet_roundtrip() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::DATA | V2Flags::DUMMY,
                log_window_size: 6,
            },
            ack: None,
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: Some(DataHeader { data_seq_num: 99 }),
            ack_vector: None,
            data_body: Some(DataBody {
                channel_seq_num: 50,
                data: vec![0x00; 10],
            }),
        };

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");

        assert!(decoded.header.flags.contains(V2Flags::DUMMY));
        assert_eq!(decoded, packet);
    }

    /// Near-MTU data packet.
    #[test]
    fn v2_large_data_packet_roundtrip() {
        let packet = V2Packet {
            header: V2Header {
                flags: V2Flags::DATA,
                log_window_size: 10,
            },
            ack: None,
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks: None,
            data_header: Some(DataHeader { data_seq_num: 1000 }),
            ack_vector: None,
            data_body: Some(DataBody {
                channel_seq_num: 500,
                data: vec![0xCC; 1200],
            }),
        };

        let encoded = encode_vec(&packet).expect("encode");
        let decoded: V2Packet = decode(&encoded).expect("decode");
        assert_eq!(decoded, packet);
    }
}
