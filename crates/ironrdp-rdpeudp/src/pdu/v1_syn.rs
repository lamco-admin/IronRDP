//! SYN-related payloads for the v1 three-way handshake.
//!
//! `RDPUDP_SYNDATA_PAYLOAD` (MS-RDPEUDP Section 2.2.2.5) and
//! `RDPUDP_SYNDATAEX_PAYLOAD` (MS-RDPEUDP Section 2.2.2.9).

use bitflags::bitflags;
use ironrdp_core::{Decode, DecodeResult, Encode, EncodeResult, ReadCursor, WriteCursor};

// -- MTU constants per MS-RDPEUDP Section 2.2.2.5 --

/// Minimum allowed MTU value.
pub const MTU_MIN: u16 = 1132;

/// Maximum allowed MTU value.
pub const MTU_MAX: u16 = 1232;

// -- RDPUDP_SYNDATA_PAYLOAD --

/// Connection initialization parameters exchanged in SYN and SYN+ACK datagrams.
///
/// MS-RDPEUDP Section 2.2.2.5.
/// Wire layout: `snInitialSequenceNumber(4)` + `uUpStreamMtu(2)` + `uDownStreamMtu(2)` = 8 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynDataPayload {
    /// Starting sequence number (random, analogous to TCP ISN per RFC 1948).
    pub initial_sequence_number: u32,

    /// Maximum datagram size this endpoint can generate.
    /// Must be in `1132..=1232`.
    pub upstream_mtu: u16,

    /// Maximum datagram size this endpoint can accept.
    /// Must be in `1132..=1232`.
    pub downstream_mtu: u16,
}

impl SynDataPayload {
    const FIXED_PART_SIZE: usize = 4 + 2 + 2;
    const NAME: &'static str = "RDPUDP_SYNDATA_PAYLOAD";

    fn validate_mtu(value: u16, field: &'static str) -> DecodeResult<()> {
        if !(MTU_MIN..=MTU_MAX).contains(&value) {
            return Err(ironrdp_core::invalid_field_err!(
                "RDPUDP_SYNDATA_PAYLOAD",
                field,
                "mtu must be in 1132..=1232"
            ));
        }
        Ok(())
    }
}

impl Encode for SynDataPayload {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_fixed_part_size!(in: dst);

        dst.write_u32(self.initial_sequence_number);
        dst.write_u16(self.upstream_mtu);
        dst.write_u16(self.downstream_mtu);

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE
    }
}

impl Decode<'_> for SynDataPayload {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_fixed_part_size!(in: src);

        let initial_sequence_number = src.read_u32();
        let upstream_mtu = src.read_u16();
        let downstream_mtu = src.read_u16();

        Self::validate_mtu(upstream_mtu, "uUpStreamMtu")?;
        Self::validate_mtu(downstream_mtu, "uDownStreamMtu")?;

        Ok(Self {
            initial_sequence_number,
            upstream_mtu,
            downstream_mtu,
        })
    }
}

// -- SYNDATAEX flags --

bitflags! {
    /// Flags in the RDPUDP_SYNDATAEX_PAYLOAD `uSynExFlags` field.
    ///
    /// MS-RDPEUDP Section 2.2.2.9.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct SynExFlags: u16 {
        /// The `uUdpVer` field indicates a supported protocol version.
        const VERSION_INFO_VALID = 0x0001;
    }
}

// -- UDP protocol versions --

/// RDPEUDP protocol version negotiated via SYNDATAEX.
///
/// MS-RDPEUDP Section 2.2.2.9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum UdpVersion {
    /// v1: min retransmit 500ms, min ACK delay 200ms.
    V1 = 0x0001,
    /// v2: min retransmit 300ms, min ACK delay 50ms.
    /// Data transfer uses MS-RDPEUDP2 wire format.
    V2 = 0x0002,
    /// v3: same as v2, plus delay-based congestion control.
    /// SYN includes cookieHash for security binding.
    V3 = 0x0101,
}

impl UdpVersion {
    /// Try to parse a version from its wire representation.
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x0001 => Some(Self::V1),
            0x0002 => Some(Self::V2),
            0x0101 => Some(Self::V3),
            _ => None,
        }
    }

    /// Returns the wire representation.
    pub fn as_u16(self) -> u16 {
        match self {
            Self::V1 => 0x0001,
            Self::V2 => 0x0002,
            Self::V3 => 0x0101,
        }
    }

    /// Whether this version uses the RDPEUDP2 wire format for data transfer.
    pub fn uses_v2_wire_format(self) -> bool {
        matches!(self, Self::V2 | Self::V3)
    }

    /// Minimum retransmission timeout for this version.
    pub fn min_retransmit_ms(self) -> u32 {
        match self {
            Self::V1 => 500,
            Self::V2 | Self::V3 => 300,
        }
    }

    /// Minimum delayed ACK timeout for this version.
    pub fn min_ack_delay_ms(self) -> u32 {
        match self {
            Self::V1 => 200,
            Self::V2 | Self::V3 => 50,
        }
    }
}

// -- RDPUDP_SYNDATAEX_PAYLOAD --

/// Extended SYN parameters for version negotiation.
///
/// MS-RDPEUDP Section 2.2.2.9.
/// Wire layout: `uSynExFlags(2)` + `uUdpVer(2)` + optional `cookieHash(32)`.
///
/// `cookieHash` is present only in a client→server SYN with v3 (`RDPUDP_PROTOCOL_VERSION_3`).
/// It contains the SHA-256 hash of the server's `securityCookie` from the
/// Initiate Multitransport Request PDU (MS-RDPBCGR Section 2.2.15.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynDataExPayload {
    /// Extended flags (must include `VERSION_INFO_VALID` for version negotiation).
    pub syn_ex_flags: SynExFlags,

    /// Protocol version advertised by this endpoint.
    pub udp_ver: UdpVersion,

    /// SHA-256 hash of securityCookie. Present only for v3 client→server SYN.
    /// Encoded as 8 × 4-byte big-endian unsigned integers.
    pub cookie_hash: Option<[u8; 32]>,
}

impl SynDataExPayload {
    const FIXED_PART_SIZE: usize = 2 + 2; // uSynExFlags + uUdpVer
    const COOKIE_HASH_SIZE: usize = 32;
    const NAME: &'static str = "RDPUDP_SYNDATAEX_PAYLOAD";

    /// Total encoded size including optional cookie hash.
    fn encoded_size(&self) -> usize {
        let base = Self::FIXED_PART_SIZE;
        if self.cookie_hash.is_some() {
            base + Self::COOKIE_HASH_SIZE
        } else {
            base
        }
    }
}

impl Encode for SynDataExPayload {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ironrdp_core::ensure_size!(in: dst, size: self.encoded_size());

        dst.write_u16(self.syn_ex_flags.bits());
        dst.write_u16(self.udp_ver.as_u16());

        if let Some(hash) = &self.cookie_hash {
            dst.write_slice(hash);
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        self.encoded_size()
    }
}

impl Decode<'_> for SynDataExPayload {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ironrdp_core::ensure_fixed_part_size!(in: src);

        let syn_ex_flags_raw = src.read_u16();
        let syn_ex_flags = SynExFlags::from_bits_truncate(syn_ex_flags_raw);

        let udp_ver_raw = src.read_u16();
        let udp_ver = UdpVersion::from_u16(udp_ver_raw).ok_or_else(|| {
            ironrdp_core::invalid_field_err!("RDPUDP_SYNDATAEX_PAYLOAD", "uUdpVer", "unknown protocol version")
        })?;

        // cookieHash is present only for v3 and when there are enough remaining bytes.
        // Per spec: MUST be present in client→server SYN with v3, MUST NOT be present otherwise.
        // We detect its presence by checking for remaining bytes because the caller may not
        // know the role (client vs server) at the PDU layer.
        let cookie_hash = if udp_ver == UdpVersion::V3 && src.len() >= Self::COOKIE_HASH_SIZE {
            Some(src.read_array::<32>())
        } else {
            None
        };

        Ok(Self {
            syn_ex_flags,
            udp_ver,
            cookie_hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use ironrdp_core::{decode, encode_vec};

    use super::*;

    // -- SynDataPayload tests --

    const SYNDATA_BYTES: [u8; 8] = [
        0x78, 0x56, 0x34, 0x12, // snInitialSequenceNumber = 0x12345678
        0xD0, 0x04, // uUpStreamMtu = 1232
        0x6C, 0x04, // uDownStreamMtu = 1132
    ];

    fn syndata() -> SynDataPayload {
        SynDataPayload {
            initial_sequence_number: 0x1234_5678,
            upstream_mtu: 1232,
            downstream_mtu: 1132,
        }
    }

    #[test]
    fn encode_syndata() {
        let encoded = encode_vec(&syndata()).expect("encode");
        assert_eq!(encoded.as_slice(), &SYNDATA_BYTES);
    }

    #[test]
    fn decode_syndata() {
        let decoded: SynDataPayload = decode(&SYNDATA_BYTES).expect("decode");
        assert_eq!(decoded, syndata());
    }

    #[test]
    fn syndata_roundtrip() {
        let original = syndata();
        let encoded = encode_vec(&original).expect("encode");
        let decoded: SynDataPayload = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn syndata_size() {
        assert_eq!(syndata().size(), 8);
    }

    #[test]
    fn syndata_mtu_below_minimum() {
        let mut bad = SYNDATA_BYTES;
        // Set uUpStreamMtu to 1000 (below 1132)
        bad[4] = 0xE8;
        bad[5] = 0x03;
        let result: DecodeResult<SynDataPayload> = decode(&bad);
        assert!(result.is_err());
    }

    #[test]
    fn syndata_mtu_above_maximum() {
        let mut bad = SYNDATA_BYTES;
        // Set uDownStreamMtu to 2000 (above 1232)
        bad[6] = 0xD0;
        bad[7] = 0x07;
        let result: DecodeResult<SynDataPayload> = decode(&bad);
        assert!(result.is_err());
    }

    #[test]
    fn syndata_mtu_boundary_values() {
        // Both at minimum
        let min_mtu = SynDataPayload {
            initial_sequence_number: 1,
            upstream_mtu: MTU_MIN,
            downstream_mtu: MTU_MIN,
        };
        let encoded = encode_vec(&min_mtu).expect("encode");
        let decoded: SynDataPayload = decode(&encoded).expect("decode");
        assert_eq!(min_mtu, decoded);

        // Both at maximum
        let max_mtu = SynDataPayload {
            initial_sequence_number: 1,
            upstream_mtu: MTU_MAX,
            downstream_mtu: MTU_MAX,
        };
        let encoded = encode_vec(&max_mtu).expect("encode");
        let decoded: SynDataPayload = decode(&encoded).expect("decode");
        assert_eq!(max_mtu, decoded);
    }

    // -- SynDataExPayload tests --

    #[test]
    fn encode_syndataex_v2_no_cookie() {
        let payload = SynDataExPayload {
            syn_ex_flags: SynExFlags::VERSION_INFO_VALID,
            udp_ver: UdpVersion::V2,
            cookie_hash: None,
        };
        let encoded = encode_vec(&payload).expect("encode");
        assert_eq!(
            encoded.as_slice(),
            &[
                0x01, 0x00, // uSynExFlags = VERSION_INFO_VALID
                0x02, 0x00, // uUdpVer = V2
            ]
        );
        assert_eq!(payload.size(), 4);
    }

    #[test]
    fn decode_syndataex_v2_no_cookie() {
        let bytes = [0x01, 0x00, 0x02, 0x00];
        let decoded: SynDataExPayload = decode(&bytes).expect("decode");
        assert_eq!(decoded.udp_ver, UdpVersion::V2);
        assert!(decoded.cookie_hash.is_none());
    }

    #[test]
    fn encode_syndataex_v3_with_cookie() {
        let mut cookie = [0u8; 32];
        for (i, byte) in cookie.iter_mut().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            {
                *byte = i as u8;
            }
        }

        let payload = SynDataExPayload {
            syn_ex_flags: SynExFlags::VERSION_INFO_VALID,
            udp_ver: UdpVersion::V3,
            cookie_hash: Some(cookie),
        };
        let encoded = encode_vec(&payload).expect("encode");
        assert_eq!(payload.size(), 36); // 4 + 32
        assert_eq!(encoded.len(), 36);

        // Verify cookie hash bytes
        assert_eq!(&encoded[4..36], &cookie);
    }

    #[test]
    fn decode_syndataex_v3_with_cookie() {
        let mut bytes = vec![
            0x01, 0x00, // VERSION_INFO_VALID
            0x01, 0x01, // V3 = 0x0101
        ];
        let cookie: Vec<u8> = (0..32).collect();
        bytes.extend_from_slice(&cookie);

        let decoded: SynDataExPayload = decode(&bytes).expect("decode");
        assert_eq!(decoded.udp_ver, UdpVersion::V3);
        let hash = decoded.cookie_hash.expect("cookie hash should be present");
        assert_eq!(&hash[..], &cookie[..]);
    }

    #[test]
    fn syndataex_roundtrip_v2() {
        let original = SynDataExPayload {
            syn_ex_flags: SynExFlags::VERSION_INFO_VALID,
            udp_ver: UdpVersion::V2,
            cookie_hash: None,
        };
        let encoded = encode_vec(&original).expect("encode");
        let decoded: SynDataExPayload = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn syndataex_roundtrip_v3_with_cookie() {
        let original = SynDataExPayload {
            syn_ex_flags: SynExFlags::VERSION_INFO_VALID,
            udp_ver: UdpVersion::V3,
            cookie_hash: Some([0xAB; 32]),
        };
        let encoded = encode_vec(&original).expect("encode");
        let decoded: SynDataExPayload = decode(&encoded).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn syndataex_unknown_version() {
        let bytes = [
            0x01, 0x00, // VERSION_INFO_VALID
            0xFF, 0xFF, // unknown version
        ];
        let result: DecodeResult<SynDataExPayload> = decode(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn syndataex_insufficient_bytes() {
        let bytes = [0x01, 0x00]; // only 2 bytes, need 4
        let result: DecodeResult<SynDataExPayload> = decode(&bytes);
        assert!(result.is_err());
    }

    // -- UdpVersion tests --

    #[test]
    fn version_wire_values() {
        assert_eq!(UdpVersion::V1.as_u16(), 0x0001);
        assert_eq!(UdpVersion::V2.as_u16(), 0x0002);
        assert_eq!(UdpVersion::V3.as_u16(), 0x0101);
    }

    #[test]
    fn version_v2_wire_format() {
        assert!(!UdpVersion::V1.uses_v2_wire_format());
        assert!(UdpVersion::V2.uses_v2_wire_format());
        assert!(UdpVersion::V3.uses_v2_wire_format());
    }

    #[test]
    fn version_timer_minimums() {
        assert_eq!(UdpVersion::V1.min_retransmit_ms(), 500);
        assert_eq!(UdpVersion::V2.min_retransmit_ms(), 300);
        assert_eq!(UdpVersion::V3.min_retransmit_ms(), 300);

        assert_eq!(UdpVersion::V1.min_ack_delay_ms(), 200);
        assert_eq!(UdpVersion::V2.min_ack_delay_ms(), 50);
        assert_eq!(UdpVersion::V3.min_ack_delay_ms(), 50);
    }
}
