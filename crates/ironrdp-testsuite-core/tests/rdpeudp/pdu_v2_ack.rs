use ironrdp_core::{DecodeResult, Encode as _, decode, encode_vec};
use ironrdp_rdpeudp::pdu::*;
// ── AckPayload tests ──

#[test]
fn ack_payload_no_delayed() {
    let ack = AckPayload {
        seq_num: 0x1234,
        received_ts: 0x00AB_CDEF,
        send_ack_time_gap: 5,
        num_delayed_acks: 0,
        delay_ack_time_scale: 0,
        delay_ack_time_additions: Vec::new(),
    };
    let encoded = encode_vec(&ack).expect("encode");
    assert_eq!(ack.size(), 7);
    assert_eq!(encoded.len(), 7);

    let decoded: AckPayload = decode(&encoded).expect("decode");
    assert_eq!(decoded, ack);
}

#[test]
fn ack_payload_with_delayed() {
    let ack = AckPayload {
        seq_num: 0x0042,
        received_ts: 0x00_123456,
        send_ack_time_gap: 10,
        num_delayed_acks: 3,
        delay_ack_time_scale: 2,
        delay_ack_time_additions: vec![15, 20, 25],
    };
    let encoded = encode_vec(&ack).expect("encode");
    assert_eq!(ack.size(), 10); // 7 + 3
    assert_eq!(encoded.len(), 10);

    let decoded: AckPayload = decode(&encoded).expect("decode");
    assert_eq!(decoded, ack);
}

#[test]
fn ack_payload_timestamp_24bit() {
    let ack = AckPayload {
        seq_num: 0,
        received_ts: 0x00FF_FFFF, // max 24-bit value
        send_ack_time_gap: 0,
        num_delayed_acks: 0,
        delay_ack_time_scale: 0,
        delay_ack_time_additions: Vec::new(),
    };
    let encoded = encode_vec(&ack).expect("encode");
    let decoded: AckPayload = decode(&encoded).expect("decode");
    assert_eq!(decoded.received_ts, 0x00FF_FFFF);
}

#[test]
fn ack_payload_timestamp_masked() {
    // Setting bits above 24 should be masked on encode
    let ack = AckPayload {
        seq_num: 0,
        received_ts: 0xFFFF_FFFF, // bits above 24 set
        send_ack_time_gap: 0,
        num_delayed_acks: 0,
        delay_ack_time_scale: 0,
        delay_ack_time_additions: Vec::new(),
    };
    let encoded = encode_vec(&ack).expect("encode");
    let decoded: AckPayload = decode(&encoded).expect("decode");
    assert_eq!(decoded.received_ts, 0x00FF_FFFF);
}

#[test]
fn ack_payload_nibble_packing() {
    // Verify the 4-bit nibble packing
    let ack = AckPayload {
        seq_num: 0,
        received_ts: 0,
        send_ack_time_gap: 0,
        num_delayed_acks: 15,    // max for 4 bits
        delay_ack_time_scale: 8, // arbitrary 4-bit value
        delay_ack_time_additions: vec![0; 15],
    };
    let encoded = encode_vec(&ack).expect("encode");
    // The packed byte should be: (15 << 4) | 8 = 0xF8
    assert_eq!(encoded[6], 0xF8);

    let decoded: AckPayload = decode(&encoded).expect("decode");
    assert_eq!(decoded.num_delayed_acks, 15);
    assert_eq!(decoded.delay_ack_time_scale, 8);
}

#[test]
fn ack_payload_insufficient_bytes() {
    let bytes = [0x00, 0x00, 0x00]; // only 3 bytes, need 7
    let result: DecodeResult<AckPayload> = decode(&bytes);
    assert!(result.is_err());
}

#[test]
fn ack_payload_insufficient_additions() {
    // Claims 5 delayed acks but no addition bytes follow
    let bytes = [
        0x00, 0x00, // seq_num
        0x00, 0x00, 0x00, // received_ts
        0x00, // send_ack_time_gap
        0x50, // numDelayedAcks=5, scale=0
              // missing 5 addition bytes
    ];
    let result: DecodeResult<AckPayload> = decode(&bytes);
    assert!(result.is_err());
}

// ── AckVectorEntry tests ──

#[test]
fn state_map_entry() {
    let entry = AckVectorEntry::StateMap { bitmap: 0b0110_1010 };
    let byte = entry.to_byte();
    assert_eq!(byte & 0x80, 0, "MSB should be 0 for state-map");
    assert_eq!(byte, 0b0110_1010);
    let decoded = AckVectorEntry::from_byte(byte);
    assert_eq!(decoded, entry);
}

#[test]
fn run_length_received() {
    let entry = AckVectorEntry::RunLength {
        received: true,
        length: 42,
    };
    let byte = entry.to_byte();
    assert_eq!(byte, 0x80 | 0x40 | 42); // MSB=1, state=1, length=42
    let decoded = AckVectorEntry::from_byte(byte);
    assert_eq!(decoded, entry);
}

#[test]
fn run_length_not_received() {
    let entry = AckVectorEntry::RunLength {
        received: false,
        length: 7,
    };
    let byte = entry.to_byte();
    assert_eq!(byte, 0x80 | 7); // MSB=1, state=0, length=7
    let decoded = AckVectorEntry::from_byte(byte);
    assert_eq!(decoded, entry);
}

#[test]
fn run_length_max() {
    let entry = AckVectorEntry::RunLength {
        received: true,
        length: 63, // 6-bit max
    };
    let byte = entry.to_byte();
    assert_eq!(byte, 0xFF); // 0x80 | 0x40 | 0x3F
    let decoded = AckVectorEntry::from_byte(byte);
    assert_eq!(decoded, entry);
}

#[test]
fn entry_coverage() {
    let map = AckVectorEntry::StateMap { bitmap: 0x7F };
    assert_eq!(map.coverage(), 7);

    let run = AckVectorEntry::RunLength {
        received: true,
        length: 30,
    };
    assert_eq!(run.coverage(), 30);
}

// ── AckVectorPayload tests ──

#[test]
fn ack_vector_no_timestamp() {
    let payload = AckVectorPayload {
        base_seq_num: 0x0100,
        timestamp: None,
        send_ack_time_gap_ms: None,
        entries: vec![
            AckVectorEntry::RunLength {
                received: true,
                length: 10,
            },
            AckVectorEntry::StateMap { bitmap: 0b010_1010 },
        ],
    };
    let encoded = encode_vec(&payload).expect("encode");
    assert_eq!(payload.size(), 3 + 2); // 3 fixed + 2 entries
    assert_eq!(encoded.len(), 5);

    let decoded: AckVectorPayload = decode(&encoded).expect("decode");
    assert_eq!(decoded, payload);
}

#[test]
fn ack_vector_with_timestamp() {
    let payload = AckVectorPayload {
        base_seq_num: 0x0042,
        timestamp: Some(0x00AB_CDEF),
        send_ack_time_gap_ms: Some(25),
        entries: vec![AckVectorEntry::RunLength {
            received: true,
            length: 20,
        }],
    };
    let encoded = encode_vec(&payload).expect("encode");
    assert_eq!(payload.size(), 3 + 4 + 1); // 3 fixed + 4 timestamp block + 1 entry
    assert_eq!(encoded.len(), 8);

    let decoded: AckVectorPayload = decode(&encoded).expect("decode");
    assert_eq!(decoded, payload);
}

#[test]
fn ack_vector_empty_entries() {
    let payload = AckVectorPayload {
        base_seq_num: 0,
        timestamp: None,
        send_ack_time_gap_ms: None,
        entries: Vec::new(),
    };
    let encoded = encode_vec(&payload).expect("encode");
    assert_eq!(payload.size(), 3);

    let decoded: AckVectorPayload = decode(&encoded).expect("decode");
    assert_eq!(decoded, payload);
}

#[test]
fn ack_vector_roundtrip_mixed_entries() {
    let payload = AckVectorPayload {
        base_seq_num: 500,
        timestamp: Some(0x00_AABBCC),
        send_ack_time_gap_ms: Some(100),
        entries: vec![
            AckVectorEntry::RunLength {
                received: true,
                length: 63,
            },
            AckVectorEntry::StateMap { bitmap: 0b0111_1111 },
            AckVectorEntry::RunLength {
                received: false,
                length: 1,
            },
            AckVectorEntry::RunLength {
                received: true,
                length: 5,
            },
        ],
    };
    let encoded = encode_vec(&payload).expect("encode");
    let decoded: AckVectorPayload = decode(&encoded).expect("decode");
    assert_eq!(decoded, payload);
}

#[test]
fn ack_vector_insufficient_bytes() {
    let bytes = [0x00, 0x00]; // only 2 bytes, need 3
    let result: DecodeResult<AckVectorPayload> = decode(&bytes);
    assert!(result.is_err());
}

#[test]
fn ack_vector_insufficient_entries() {
    // Claims 10 entries but provides none
    let bytes = [
        0x00, 0x00, // base_seq_num
        10,   // codedAckVecSize=10, TimeStampPresent=0
    ];
    let result: DecodeResult<AckVectorPayload> = decode(&bytes);
    assert!(result.is_err());
}

#[test]
fn ack_vector_insufficient_timestamp_block() {
    // TimeStampPresent=1 but not enough bytes for the timestamp block
    let bytes = [
        0x00, 0x00, // base_seq_num
        0x80, // codedAckVecSize=0, TimeStampPresent=1
        0x05, // only 1 byte of the 4-byte timestamp block
    ];
    let result: DecodeResult<AckVectorPayload> = decode(&bytes);
    assert!(result.is_err());
}
