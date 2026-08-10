use ironrdp_core::{DecodeResult, Encode as _, decode, encode_vec};
use ironrdp_rdpeudp::pdu::*;
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

/// Server SYN+ACK: header + SYNDATA + SYNDATAEX.
///
/// No ACK vector, despite the ACK flag: see [MS-RDPEUDP] 3.1.5.1.3.
#[test]
fn v1_syn_ack_datagram_roundtrip() {
    let datagram = V1Datagram {
        header: FecHeader {
            sn_source_ack: 100,
            receive_window_size: 64,
            flags: V1Flags::SYN | V1Flags::ACK | V1Flags::SYNEX,
        },
        ack_vector: None,
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

    // header(8) + syndata(8) + syndataex(4) = 20.
    assert_eq!(datagram.size(), 20);

    let encoded = encode_vec(&datagram).expect("encode");
    let decoded: V1Datagram = decode(&encoded).expect("decode");
    assert_eq!(decoded, datagram);
    assert!(decoded.header.flags.contains(V1Flags::ACK));
}

/// The SYN+ACK capture from [MS-RDPEUDP] 4.1.2, decoded whole.
///
/// The ACK flag is set and the SYNDATA payload starts immediately after the
/// 8-byte header, so a decoder that reads an ACK vector here consumes the
/// first half of SYNDATA and desynchronises for the rest of the datagram.
#[test]
fn v1_decode_the_spec_syn_ack_capture() {
    // Trailing zeroes are the start of the pad to uUpStreamMtu.
    const CAPTURE: [u8; 19] = [
        0x00, 0x00, 0x00, 0x42, 0x04, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x42, 0x04, 0xD0, 0x04, 0xD0, 0x00, 0x00,
        0x00,
    ];

    let datagram: V1Datagram = decode(&CAPTURE).expect("decode");

    assert_eq!(datagram.header.sn_source_ack, 0x42);
    assert_eq!(datagram.header.receive_window_size, 1024);
    assert_eq!(datagram.header.flags, V1Flags::SYN | V1Flags::ACK);
    assert!(datagram.ack_vector.is_none());

    let syn_data = datagram.syn_data.expect("SYNDATA");
    assert_eq!(syn_data.initial_sequence_number, 0x42);
    assert_eq!(syn_data.upstream_mtu, 1232);
    assert_eq!(syn_data.downstream_mtu, 1232);

    assert!(datagram.syn_data_ex.is_none());
}

/// The ACK capture from [MS-RDPEUDP] 4.2.3, whose vector we do decode.
///
/// The counterpart to the SYN+ACK above: without SYN, the ACK flag means what
/// 2.2.2.1 says it means. The DATA payload is left off, since v1 data transfer
/// is out of scope for this crate.
#[test]
fn v1_decode_the_spec_ack_capture() {
    const CAPTURE: [u8; 16] = [
        // FEC header: snSourceAck, uReceiveWindowSize 1024, uFlags 0x0104.
        0xD6, 0xCF, 0x0A, 0xB8, 0x04, 0x00, 0x01, 0x04,
        // ACK vector: one element, 4 datagrams received, then a pad byte.
        0x00, 0x01, 0x04, 0x00, //
        // AckOfAcks.
        0xD6, 0xCF, 0x0A, 0xB8,
    ];

    let datagram: V1Datagram = decode(&CAPTURE).expect("decode");

    assert_eq!(datagram.header.sn_source_ack, 0xD6CF_0AB8);
    assert_eq!(datagram.header.receive_window_size, 1024);

    let ack_vector = datagram.ack_vector.as_ref().expect("ACK vector");
    assert_eq!(
        ack_vector.elements,
        vec![V1AckVectorElement {
            state: VectorElementState::DatagramReceived,
            length: 4,
        }]
    );

    assert_eq!(
        datagram.ack_of_acks.as_ref().expect("AckOfAcks").reset_seq_num,
        0xD6CF_0AB8
    );

    assert_eq!(encode_vec(&datagram).expect("encode"), CAPTURE);
}

/// A SYN datagram carrying an ACK vector is rejected rather than written.
#[test]
fn v1_encode_rejects_ack_vector_on_a_syn() {
    let datagram = V1Datagram {
        header: FecHeader {
            sn_source_ack: 100,
            receive_window_size: 64,
            flags: V1Flags::SYN | V1Flags::ACK,
        },
        ack_vector: Some(V1AckVectorHeader {
            elements: vec![V1AckVectorElement {
                state: VectorElementState::DatagramReceived,
                length: 1,
            }],
        }),
        ack_of_acks: None,
        syn_data: Some(SynDataPayload {
            initial_sequence_number: 1,
            upstream_mtu: 1232,
            downstream_mtu: 1232,
        }),
        correlation_id: None,
        syn_data_ex: None,
    };

    encode_vec(&datagram).expect_err("a SYN datagram cannot carry an ACK vector");
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
                    state: VectorElementState::DatagramReceived,
                    length: 5,
                },
                V1AckVectorElement {
                    state: VectorElementState::DatagramNotYetReceived,
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

    // header(8) + syndata(8) + correlation(16 id + 16 reserved = 32) + syndataex(4) = 52.
    // [MS-RDPEUDP] 2.2.2.8 makes RDPUDP_CORRELATION_ID_PAYLOAD 32 bytes.
    assert_eq!(datagram.size(), 52);

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
            // Caller set neither ACK nor ACK_OF_ACKS, but the payloads
            // those flags gate are populated.
            flags: V1Flags::empty(),
        },
        ack_vector: Some(V1AckVectorHeader { elements: vec![] }),
        ack_of_acks: Some(V1AckOfAcksHeader { reset_seq_num: 7 }),
        syn_data: None,
        correlation_id: None,
        // syn_data_ex is None, so SYNEX should NOT be in flags
        syn_data_ex: None,
    };

    let encoded = encode_vec(&datagram).expect("encode");
    let decoded: V1Datagram = decode(&encoded).expect("decode");

    // ACK should be auto-added (ack_vector is Some)
    assert!(decoded.header.flags.contains(V1Flags::ACK));
    // ACK_OF_ACKS should be auto-added (ack_of_acks is Some)
    assert!(decoded.header.flags.contains(V1Flags::ACK_OF_ACKS));
    // SYNEX should NOT be set (syn_data_ex is None)
    assert!(!decoded.header.flags.contains(V1Flags::SYNEX));
}

/// The ACK flag survives on a SYN+ACK, where no payload implies it.
#[test]
fn v1_ack_flag_preserved_on_a_syn() {
    let datagram = V1Datagram {
        header: FecHeader {
            sn_source_ack: 100,
            receive_window_size: 64,
            flags: V1Flags::ACK,
        },
        ack_vector: None,
        ack_of_acks: None,
        syn_data: Some(SynDataPayload {
            initial_sequence_number: 1,
            upstream_mtu: 1232,
            downstream_mtu: 1232,
        }),
        correlation_id: None,
        syn_data_ex: None,
    };

    let encoded = encode_vec(&datagram).expect("encode");
    let decoded: V1Datagram = decode(&encoded).expect("decode");

    assert_eq!(decoded.header.flags, V1Flags::SYN | V1Flags::ACK);
    assert!(decoded.ack_vector.is_none());
}

/// A plain SYN does not acquire the ACK flag.
#[test]
fn v1_ack_flag_absent_on_a_bare_syn() {
    let datagram = V1Datagram {
        header: FecHeader {
            sn_source_ack: 0xFFFF_FFFF,
            receive_window_size: 64,
            flags: V1Flags::empty(),
        },
        ack_vector: None,
        ack_of_acks: None,
        syn_data: Some(SynDataPayload {
            initial_sequence_number: 1,
            upstream_mtu: 1232,
            downstream_mtu: 1232,
        }),
        correlation_id: None,
        syn_data_ex: None,
    };

    let encoded = encode_vec(&datagram).expect("encode");
    let decoded: V1Datagram = decode(&encoded).expect("decode");

    assert_eq!(decoded.header.flags, V1Flags::SYN);
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
                state: VectorElementState::DatagramReceived,
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
    let [low, high] = modified.to_le_bytes();
    encoded[6] = low;
    encoded[7] = high;

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

/// The v2 header has no standalone flags, so a bit the caller sets without the
/// payload it announces does not survive the round trip.
///
/// Every flag [MS-RDPEUDP2] 2.2.1.1 defines announces a payload. There is
/// nothing like the v1 header's CN, CWR or SYNLOSSY, which stand on their own
/// and have to be carried across.
#[test]
fn v2_has_no_standalone_flags() {
    let packet = V2Packet {
        header: V2Header {
            // ACKVEC announces a payload this packet does not carry.
            flags: V2Flags::ACK | V2Flags::ACKVEC,
            log_window_size: 6,
        },
        ack: Some(AckPayload {
            seq_num: 1,
            received_ts: 0,
            send_ack_time_gap: 0,
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

    assert_eq!(
        decoded.header.flags,
        V2Flags::ACK,
        "flags should describe exactly the payloads present"
    );
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

/// The v2 header defines exactly six flags.
///
/// [MS-RDPEUDP2] 2.2.1.1 lists ACK, DATA, ACKVEC, AOA, OVERHEADSIZE and
/// DELAYACKINFO, and nothing else. In particular there is no DUMMY flag: a
/// dummy packet is marked by Packet_Type_Index 8 in the PacketPrefixByte
/// (3.1.1.1.5), one layer down.
#[test]
fn v2_header_flags_match_the_spec_table() {
    assert_eq!(V2Flags::ACK.bits(), 0x001);
    assert_eq!(V2Flags::DATA.bits(), 0x004);
    assert_eq!(V2Flags::ACKVEC.bits(), 0x008);
    assert_eq!(V2Flags::AOA.bits(), 0x010);
    assert_eq!(V2Flags::OVERHEADSIZE.bits(), 0x040);
    assert_eq!(V2Flags::DELAYACKINFO.bits(), 0x100);
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
