use core::time::Duration;
use ironrdp_core::decode;
use ironrdp_rdpeudp::pdu::*;
use ironrdp_rdpeudp::*;
fn now() -> MonotonicInstant {
    MonotonicInstant::from_millis(0)
}

fn later(base: MonotonicInstant, ms: u64) -> MonotonicInstant {
    base + Duration::from_millis(ms)
}

fn default_config(isn: u32) -> ConnectionConfig {
    ConnectionConfig {
        initial_sequence_number: isn,
        log_window_size: 6,
        upstream_mtu: 1232,
        downstream_mtu: 1232,
        idle_timeout: Duration::from_secs(16),
        keep_alive_interval: Duration::from_secs(8),
        ack_delay_timeout: Duration::from_millis(50),
    }
}

// ── Handshake tests ──

#[test]
fn client_connect_produces_syn() {
    let t = now();
    let mut conn = RdpeudpConnection::connect(default_config(100), t);

    let transmit = conn.poll_transmit(t).expect("should have SYN");
    assert!(!transmit.contents.is_empty());

    // Decode the SYN
    let datagram: V1Datagram = decode(&transmit.contents).expect("valid SYN");
    assert!(datagram.header.flags.contains(V1Flags::SYN));
    assert!(datagram.header.flags.contains(V1Flags::SYNEX));
    assert_eq!(
        datagram
            .syn_data
            .as_ref()
            .expect("has syn_data")
            .initial_sequence_number,
        100
    );

    // No more transmits
    assert!(conn.poll_transmit(t).is_none());

    // Should be in SynSent state
    assert!(!conn.is_established());
    assert!(!conn.is_closed());
}

#[test]
fn server_accept_produces_syn_ack() {
    let t = now();

    // Build a client SYN
    let client_syn = V1Datagram {
        header: FecHeader {
            sn_source_ack: 0xFFFF_FFFF,
            receive_window_size: 64,
            flags: V1Flags::SYN | V1Flags::SYNEX,
        },
        ack_vector: None,
        ack_of_acks: None,
        syn_data: Some(SynDataPayload {
            initial_sequence_number: 100,
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

    let mut conn = RdpeudpConnection::accept(default_config(200), &client_syn, t).expect("accept");

    let transmit = conn.poll_transmit(t).expect("should have SYN+ACK");

    // Decode the SYN+ACK
    let datagram: V1Datagram = decode(&transmit.contents).expect("valid SYN+ACK");
    assert!(datagram.header.flags.contains(V1Flags::SYN));
    assert!(datagram.header.flags.contains(V1Flags::ACK));
    assert!(datagram.header.flags.contains(V1Flags::SYNEX));
    assert_eq!(
        datagram
            .syn_data
            .as_ref()
            .expect("has syn_data")
            .initial_sequence_number,
        200
    );
}

#[test]
fn full_handshake_client_server() {
    let t = now();

    // Step 1: Client sends SYN
    let mut client = RdpeudpConnection::connect(default_config(100), t);
    let syn_transmit = client.poll_transmit(t).expect("client SYN");

    // Step 2: Server receives SYN and creates connection with SYN+ACK
    let syn_datagram: V1Datagram = decode(&syn_transmit.contents).expect("decode SYN");
    let mut server = RdpeudpConnection::accept(default_config(200), &syn_datagram, t).expect("accept");
    let syn_ack_transmit = server.poll_transmit(t).expect("server SYN+ACK");

    // Step 3: Client receives SYN+ACK → transitions to Established
    let mut syn_ack_bytes = syn_ack_transmit.contents;
    client
        .handle_datagram(&mut syn_ack_bytes, later(t, 50))
        .expect("handle SYN+ACK");

    assert!(client.is_established());

    // Client should emit Connected event
    let event = client.poll_event().expect("should have event");
    assert_eq!(event, Event::Connected);

    // Client sends final ACK
    let ack_transmit = client.poll_transmit(later(t, 50)).expect("client final ACK");

    // Step 4: Server receives final ACK → transitions to Established
    let mut ack_bytes = ack_transmit.contents;
    server
        .handle_datagram(&mut ack_bytes, later(t, 100))
        .expect("handle final ACK");

    assert!(server.is_established());

    let event = server.poll_event().expect("should have event");
    assert_eq!(event, Event::Connected);
}

#[test]
fn server_rejects_non_v2_syn() {
    let t = now();

    let syn = V1Datagram {
        header: FecHeader {
            sn_source_ack: 0xFFFF_FFFF,
            receive_window_size: 64,
            flags: V1Flags::SYN | V1Flags::SYNEX,
        },
        ack_vector: None,
        ack_of_acks: None,
        syn_data: Some(SynDataPayload {
            initial_sequence_number: 100,
            upstream_mtu: 1232,
            downstream_mtu: 1232,
        }),
        correlation_id: None,
        syn_data_ex: Some(SynDataExPayload {
            syn_ex_flags: SynExFlags::VERSION_INFO_VALID,
            udp_ver: UdpVersion::V1,
            cookie_hash: None,
        }),
    };

    let result = RdpeudpConnection::accept(default_config(200), &syn, t);
    assert!(result.is_err());
}

#[test]
fn send_before_established_returns_error() {
    let t = now();
    let mut conn = RdpeudpConnection::connect(default_config(100), t);

    let result = conn.send(vec![0xAA; 100]);
    assert!(matches!(result.unwrap_err().kind(), RdpeudpErrorKind::InvalidState));
}

// ── Data transfer tests ──

/// Helper: perform a full handshake and return (client, server, time).
fn establish_pair() -> (RdpeudpConnection, RdpeudpConnection, MonotonicInstant) {
    let t = now();

    let mut client = RdpeudpConnection::connect(default_config(100), t);
    let syn = client.poll_transmit(t).expect("SYN");

    let syn_dg: V1Datagram = decode(&syn.contents).expect("decode SYN");
    let mut server = RdpeudpConnection::accept(default_config(200), &syn_dg, t).expect("accept");
    let syn_ack = server.poll_transmit(t).expect("SYN+ACK");

    let mut syn_ack_bytes = syn_ack.contents;
    client
        .handle_datagram(&mut syn_ack_bytes, later(t, 50))
        .expect("handle SYN+ACK");

    // Drain client events and final ACK
    while client.poll_event().is_some() {}
    let final_ack = client.poll_transmit(later(t, 50)).expect("final ACK");

    let mut ack_bytes = final_ack.contents;
    server
        .handle_datagram(&mut ack_bytes, later(t, 100))
        .expect("handle ACK");
    while server.poll_event().is_some() {}

    (client, server, t)
}

#[test]
fn send_data_after_established() {
    let (mut client, _server, t) = establish_pair();

    // Send data
    client.send(vec![0xDE, 0xAD, 0xBE, 0xEF]).expect("send");

    // Should produce a data transmit
    let transmit = client.poll_transmit(later(t, 200)).expect("data packet");
    assert!(!transmit.contents.is_empty());
}

#[test]
fn data_delivery_roundtrip() {
    let (mut client, mut server, t) = establish_pair();

    // Client sends data
    let payload = vec![0xDE, 0xAD, 0xBE, 0xEF];
    client.send(payload.clone()).expect("send");
    let data_pkt = client.poll_transmit(later(t, 200)).expect("data packet");

    // Server receives data
    let mut data_bytes = data_pkt.contents;
    server
        .handle_datagram(&mut data_bytes, later(t, 250))
        .expect("handle data");

    // Server should emit DataReceived event
    let event = server.poll_event().expect("should have event");
    assert_eq!(event, Event::DataReceived(payload));
}

#[test]
fn bidirectional_data() {
    let (mut client, mut server, t) = establish_pair();

    // Client → Server
    let c2s = vec![0x01, 0x02, 0x03];
    client.send(c2s.clone()).expect("send c2s");
    let pkt = client.poll_transmit(later(t, 200)).expect("c2s packet");
    let mut pkt_bytes = pkt.contents;
    server
        .handle_datagram(&mut pkt_bytes, later(t, 250))
        .expect("handle c2s");

    let event = server.poll_event().expect("c2s event");
    assert_eq!(event, Event::DataReceived(c2s));

    // Server → Client
    let s2c = vec![0x04, 0x05, 0x06];
    server.send(s2c.clone()).expect("send s2c");
    let pkt = server.poll_transmit(later(t, 300)).expect("s2c packet");
    let mut pkt_bytes = pkt.contents;
    client
        .handle_datagram(&mut pkt_bytes, later(t, 350))
        .expect("handle s2c");

    let event = client.poll_event().expect("s2c event");
    assert_eq!(event, Event::DataReceived(s2c));
}

// ── Close and timeout tests ──

#[test]
fn close_produces_event() {
    let (mut client, _server, _t) = establish_pair();
    client.close();

    assert!(client.is_closed());
    let event = client.poll_event().expect("close event");
    assert_eq!(event, Event::ConnectionClosed);
}

#[test]
fn send_after_close_returns_error() {
    let (mut client, _server, _t) = establish_pair();
    client.close();

    let result = client.send(vec![0x42]);
    assert!(matches!(result.unwrap_err().kind(), RdpeudpErrorKind::ConnectionClosed));
}

#[test]
fn idle_timeout_closes_connection() {
    let (mut client, _server, t) = establish_pair();

    // Advance time past idle timeout
    let idle_time = later(t, 17_000); // 17 seconds > 16 second timeout
    client.handle_timeout(idle_time);

    assert!(client.is_closed());
    let event = client.poll_event().expect("idle close event");
    assert_eq!(event, Event::ConnectionClosed);
}

#[test]
fn keep_alive_fires() {
    let (mut client, _server, t) = establish_pair();

    // Advance time past keep-alive interval
    let ka_time = later(t, 8_500); // 8.5 seconds > 8 second interval
    client.handle_timeout(ka_time);

    // The keep-alive leaves an acknowledgement pending, which the next
    // poll_transmit turns into a probe on the wire.
    assert!(client.poll_transmit(ka_time).is_some());
}

#[test]
fn poll_timeout_returns_earliest() {
    let (client, _server, _t) = establish_pair();

    let timeout = client.poll_timeout();
    assert!(timeout.is_some());
}

// ── Congestion backpressure test ──

#[test]
fn congestion_backpressure() {
    let (mut client, _server, t) = establish_pair();

    // Fill the congestion window
    let mtu_data = vec![0xAA; 1200]; // near MTU
    let mut sent_count = 0u32;

    for _ in 0..100 {
        if client.send(mtu_data.clone()).is_ok() {
            sent_count += 1;
        }
    }
    assert!(sent_count > 0);

    // Drain transmits until congestion window is full
    let mut transmit_count = 0u32;
    let t2 = later(t, 200);
    while client.poll_transmit(t2).is_some() {
        transmit_count += 1;
        if transmit_count > 200 {
            break; // safety valve
        }
    }

    // Should have been limited by the congestion window
    // Initial window is 12320 bytes, each packet ~1200 bytes → ~10 packets
    assert!(transmit_count > 0);
    assert!(transmit_count <= 15); // reasonable upper bound with overhead
}

#[test]
fn multiple_data_chunks_delivered_in_order() {
    let (mut client, mut server, t) = establish_pair();

    // Send multiple data chunks
    for i in 0u8..5 {
        client.send(vec![i; 100]).expect("send");
    }

    // Transmit and deliver each
    for i in 0u8..5 {
        let pkt = client.poll_transmit(later(t, 200 + u64::from(i) * 50));
        if let Some(pkt) = pkt {
            let mut pkt_bytes = pkt.contents;
            server
                .handle_datagram(&mut pkt_bytes, later(t, 225 + u64::from(i) * 50))
                .expect("handle data");
        }
    }

    // Verify all events arrived in order
    let mut received = Vec::new();
    while let Some(event) = server.poll_event() {
        if let Event::DataReceived(data) = event {
            received.push(data[0]); // first byte identifies the chunk
        }
    }

    assert_eq!(received, vec![0, 1, 2, 3, 4]);
}

#[test]
fn handle_datagram_after_close_returns_error() {
    let (mut client, _server, t) = establish_pair();
    client.close();

    let mut wire = vec![0u8; 16];
    let result = client.handle_datagram(&mut wire, later(t, 100));
    assert!(matches!(result.unwrap_err().kind(), RdpeudpErrorKind::ConnectionClosed));
}

#[test]
fn config_default() {
    let config = ConnectionConfig::default();
    assert_eq!(config.log_window_size, 6);
    assert_eq!(config.upstream_mtu, 1232);
    assert_eq!(config.downstream_mtu, 1232);
    assert_eq!(config.idle_timeout, Duration::from_secs(16));
    assert_eq!(config.keep_alive_interval, Duration::from_secs(8));
    assert_eq!(config.ack_delay_timeout, Duration::from_millis(50));
}

#[test]
fn debug_impl() {
    let t = now();
    let conn = RdpeudpConnection::connect(default_config(1), t);
    let debug = format!("{conn:?}");
    assert!(debug.contains("RdpeudpConnection"));
    assert!(debug.contains("Client"));
}

/// Regression: the idle timeout must not leave a re-armed timer behind.
///
/// `handle_timeout` collects the expired timers, then handles them in order.
/// The idle timer closes the connection and `close` clears every timer, but a
/// handler later in the same list used to run anyway and re-arm its own timer
/// on a connection that had already gone. `handle_timeout` returns early once
/// closed, so that deadline was never serviced: a driver polling it without
/// checking `is_closed` first would wake immediately, do nothing, and spin.
///
/// Found by the `rdpeudp_connection` fuzz oracle.
#[test]
fn idle_close_leaves_no_timer_armed() {
    let mut now = MonotonicInstant::from_millis(0);
    let mut conn = RdpeudpConnection::connect(default_config(100), now);
    while conn.poll_transmit(now).is_some() {}

    // Walk past the keep-alive interval so that timer is armed and due in the
    // same pass as the idle timer, which is the ordering that used to break.
    for step in [1_000u64, 8_000, 16_000] {
        now = now + Duration::from_millis(step);
        conn.handle_timeout(now);
        while conn.poll_transmit(now).is_some() {}
    }

    assert!(conn.is_closed(), "the idle timeout should have closed the connection");
    assert_eq!(
        conn.poll_timeout(),
        None,
        "a closed connection must not report a deadline"
    );

    // Advancing again must not resurrect one.
    now = now + Duration::from_secs(60);
    conn.handle_timeout(now);
    assert_eq!(conn.poll_timeout(), None, "a closed connection re-armed a timer");
}
