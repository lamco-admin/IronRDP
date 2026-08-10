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
        idle_timeout: Duration::from_secs(65),
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
    let idle_time = later(t, 66_000); // 66 seconds > the 65 second timeout
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
    assert_eq!(config.idle_timeout, Duration::from_secs(65));
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
    for step in [1_000u64, 8_000, 65_000] {
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

/// Regression: a closed connection must not emit packets or arm timers.
///
/// `poll_transmit` drained queued handshake packets before checking the state,
/// and armed the keep-alive timer on the way past. `close` had already cleared
/// the timers and `handle_timeout` returns early once closed, so that deadline
/// stayed due forever and a caller polling it without checking `is_closed`
/// would wake immediately, do nothing, and spin.
///
/// Found by the `rdpeudp_connection` fuzz oracle.
#[test]
fn closed_connection_transmits_nothing_and_arms_nothing() {
    let now = MonotonicInstant::from_millis(0);

    // Close while the SYN is still queued, which is the state the oracle found.
    let mut conn = RdpeudpConnection::connect(default_config(100), now);
    conn.close();

    assert!(
        conn.poll_transmit(now).is_none(),
        "a closed connection emitted a packet"
    );
    assert_eq!(conn.poll_timeout(), None, "a closed connection armed a timer");

    // And it stays that way once the clock moves on.
    let later = now + Duration::from_secs(30);
    conn.handle_timeout(later);
    assert!(conn.poll_transmit(later).is_none());
    assert_eq!(conn.poll_timeout(), None);
}

// ── Handshake retransmission ──
//
// [MS-RDPEUDP] 1.3.1: the SYN, SYN+ACK and ACK are delivered by persistent
// retransmits whatever mode the transport is running in. 3.1.5.4.1 stops
// after somewhere between three and five unanswered tries.

/// Drive the connection to its next retransmit deadline and collect whatever
/// it decides to send.
fn advance_to_retransmit(conn: &mut RdpeudpConnection) -> Option<Vec<u8>> {
    let deadline = conn.poll_timeout()?;
    conn.handle_timeout(deadline);
    conn.poll_transmit(deadline).map(|transmit| transmit.contents)
}

#[test]
fn unanswered_syn_is_retransmitted() {
    let t = now();
    let mut client = RdpeudpConnection::connect(default_config(100), t);

    let first = client.poll_transmit(t).expect("SYN");
    let again = advance_to_retransmit(&mut client).expect("SYN again");

    assert_eq!(first.contents, again, "the retransmitted SYN differs from the original");
    assert!(!client.is_closed());
}

#[test]
fn unanswered_syn_ack_is_retransmitted() {
    let t = now();

    let mut client = RdpeudpConnection::connect(default_config(100), t);
    let syn = client.poll_transmit(t).expect("SYN");
    let syn_dg: V1Datagram = decode(&syn.contents).expect("decode SYN");

    let mut server = RdpeudpConnection::accept(default_config(200), &syn_dg, t).expect("accept");
    let first = server.poll_transmit(t).expect("SYN+ACK");
    let again = advance_to_retransmit(&mut server).expect("SYN+ACK again");

    assert_eq!(first.contents, again);
    assert!(!server.is_closed());
}

#[test]
fn a_syn_that_is_never_answered_closes_the_connection() {
    let t = now();
    let mut client = RdpeudpConnection::connect(default_config(100), t);
    client.poll_transmit(t).expect("SYN");

    // The retransmit timer keeps firing until the limit is reached. The idle
    // timer would also close this connection eventually, so the assertion
    // below is that we gave up on the retransmit count first.
    let mut retransmits = 0;
    while !client.is_closed() {
        let deadline = client.poll_timeout().expect("a timer is armed");
        assert!(
            deadline < t + client_idle_timeout(),
            "gave up on the idle timeout rather than the retransmit limit"
        );

        client.handle_timeout(deadline);
        if client.poll_transmit(deadline).is_some() {
            retransmits += 1;
        }
    }

    assert_eq!(retransmits, 5, "[MS-RDPEUDP] 3.1.5.4.1 allows between three and five");
    assert_eq!(client.poll_event(), Some(Event::ConnectionClosed));
}

fn client_idle_timeout() -> Duration {
    default_config(0).idle_timeout
}

#[test]
fn a_repeated_syn_draws_another_syn_ack() {
    let t = now();

    let mut client = RdpeudpConnection::connect(default_config(100), t);
    let syn = client.poll_transmit(t).expect("SYN");
    let syn_dg: V1Datagram = decode(&syn.contents).expect("decode SYN");

    let mut server = RdpeudpConnection::accept(default_config(200), &syn_dg, t).expect("accept");
    let syn_ack = server.poll_transmit(t).expect("SYN+ACK");

    // The client did not see that SYN+ACK, so it repeats its SYN.
    let mut repeated = syn.contents;
    server
        .handle_datagram(&mut repeated, later(t, 500))
        .expect("a repeated SYN is not a protocol violation");

    let answer = server.poll_transmit(later(t, 500)).expect("another SYN+ACK");
    assert_eq!(answer.contents, syn_ack.contents);
    assert!(!server.is_established());
}

#[test]
fn a_repeated_syn_ack_draws_another_final_ack() {
    let t = now();

    let mut client = RdpeudpConnection::connect(default_config(100), t);
    let syn = client.poll_transmit(t).expect("SYN");
    let syn_dg: V1Datagram = decode(&syn.contents).expect("decode SYN");

    let mut server = RdpeudpConnection::accept(default_config(200), &syn_dg, t).expect("accept");
    let syn_ack = server.poll_transmit(t).expect("SYN+ACK");

    let mut syn_ack_bytes = syn_ack.contents.clone();
    client
        .handle_datagram(&mut syn_ack_bytes, later(t, 50))
        .expect("handle SYN+ACK");
    while client.poll_event().is_some() {}

    let final_ack = client.poll_transmit(later(t, 50)).expect("final ACK");
    assert!(client.is_established());

    // That ACK was lost, so the server repeats its SYN+ACK. The client is on
    // v2 by now and must still recognise it.
    let mut repeated = syn_ack.contents;
    client
        .handle_datagram(&mut repeated, later(t, 600))
        .expect("a repeated SYN+ACK is not a protocol violation");

    let answer = client.poll_transmit(later(t, 600)).expect("the final ACK again");
    assert_eq!(answer.contents, final_ack.contents);
    assert!(client.is_established(), "the connection did not fall back a state");
}

#[test]
fn data_is_not_mistaken_for_a_repeated_syn_ack() {
    let (mut client, mut server, t) = establish_pair();

    server.send(vec![0x11; 32]).expect("send");
    let mut data = server.poll_transmit(later(t, 200)).expect("data").contents;

    client.handle_datagram(&mut data, later(t, 250)).expect("handle data");

    assert_eq!(client.poll_event(), Some(Event::DataReceived(vec![0x11; 32])));
}
