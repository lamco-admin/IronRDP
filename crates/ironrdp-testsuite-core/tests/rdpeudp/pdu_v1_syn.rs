use ironrdp_core::{DecodeResult, Encode as _, decode, encode_vec};
use ironrdp_rdpeudp::pdu::*;
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
        *byte = u8::try_from(i % 256).expect("modulo 256 fits in u8");
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
    assert_eq!(hash.as_slice(), cookie.as_slice());
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
