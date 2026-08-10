use ironrdp_core::{DecodeResult, Encode as _, decode, encode_vec};
use ironrdp_rdpeudp::pdu::*;
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
