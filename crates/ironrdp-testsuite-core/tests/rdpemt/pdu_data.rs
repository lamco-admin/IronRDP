use ironrdp_core::Encode as _;
use ironrdp_rdpemt::pdu::*;
#[test]
fn round_trip_simple_data() {
    let original = TunnelData {
        sub_headers: Vec::new(),
        higher_layer_data: vec![0x48, 0x65, 0x6C, 0x6C, 0x6F], // "Hello"
    };

    let encoded = ironrdp_core::encode_vec(&original).expect("encode");
    // Header: Action=2, PayloadLen=5, HeaderLen=4, then 5 data bytes
    assert_eq!(encoded, [0x02, 0x05, 0x00, 0x04, 0x48, 0x65, 0x6C, 0x6C, 0x6F]);

    let decoded: TunnelData = ironrdp_core::decode(&encoded).expect("decode");
    assert_eq!(decoded, original);
}

#[test]
fn round_trip_data_with_subheader() {
    let original = TunnelData {
        sub_headers: vec![TunnelSubHeader {
            sub_header_type: SubHeaderType::AutoDetectRequest,
            data: vec![0xFF],
        }],
        higher_layer_data: vec![0x01, 0x02],
    };

    let encoded = ironrdp_core::encode_vec(&original).expect("encode");
    // Header: Action=2, PayloadLen=2, HeaderLen=7 (4 + subheader(3))
    // SubHeader: len=3, type=0x00, data=0xFF
    // Payload: 0x01, 0x02
    assert_eq!(encoded, [0x02, 0x02, 0x00, 0x07, 0x03, 0x00, 0xFF, 0x01, 0x02]);

    let decoded: TunnelData = ironrdp_core::decode(&encoded).expect("decode");
    assert_eq!(decoded, original);
}

#[test]
fn round_trip_empty_data() {
    let original = TunnelData {
        sub_headers: Vec::new(),
        higher_layer_data: Vec::new(),
    };

    let encoded = ironrdp_core::encode_vec(&original).expect("encode");
    assert_eq!(encoded, [0x02, 0x00, 0x00, 0x04]);

    let decoded: TunnelData = ironrdp_core::decode(&encoded).expect("decode");
    assert_eq!(decoded, original);
}

#[test]
fn size_calculation() {
    let pdu = TunnelData {
        sub_headers: Vec::new(),
        higher_layer_data: vec![0; 100],
    };
    // 4 (header) + 100 (data) = 104
    assert_eq!(pdu.size(), 104);
}
