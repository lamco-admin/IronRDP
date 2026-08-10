//! Full-stack integration tests: client <-> server over loopback UDP.
//!
//! Each test establishes a complete RDPEUDP2 + TLS + RDPEMT tunnel
//! between a client (`connect_udp`) and server (`accept_udp`) on
//! localhost, then exercises bidirectional data transfer.
//!
//! These tests use `multi_thread` flavor because they perform real
//! UDP I/O, which conflicts with tokio's mock clock (test-util).

use std::sync::Arc;
use std::time::Duration;

use ironrdp_rdpemt::TunnelConfig;
use ironrdp_rdpeudp::ConnectionConfig;
use ironrdp_rdpeudp_tokio::{UdpAcceptConfig, UdpTransport, UdpTransportConfig, accept_udp, connect_udp};
use tokio::net::UdpSocket;

/// Generate a self-signed TLS server config for testing.
fn test_tls_server_config() -> Arc<tokio_rustls::rustls::ServerConfig> {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).expect("generate cert");

    let cert_der = tokio_rustls::rustls::pki_types::CertificateDer::from(cert.cert.der().to_vec());
    // rcgen 0.14 renamed `CertifiedKey::key_pair` to `signing_key`.
    let key_der = tokio_rustls::rustls::pki_types::PrivateKeyDer::try_from(cert.signing_key.serialize_der())
        .expect("serialize key");

    let config = tokio_rustls::rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)
        .expect("server config");

    Arc::new(config)
}

fn test_tunnel_config() -> TunnelConfig {
    TunnelConfig {
        request_id: 42,
        security_cookie: [0xAB; 16],
    }
}

/// Establish a connected client/server pair over loopback.
///
/// Returns `(client, server)`: both are `UdpTransport` handles
/// ready for bidirectional data transfer.
async fn establish_loopback_pair() -> (UdpTransport, UdpTransport) {
    let server_sock = UdpSocket::bind("127.0.0.1:0").await.expect("bind server");
    let server_addr = server_sock.local_addr().expect("server addr");

    let tunnel_config = test_tunnel_config();

    let server_handle = tokio::spawn({
        let tunnel_config = tunnel_config.clone();
        async move {
            accept_udp(
                server_sock,
                UdpAcceptConfig {
                    tls_config: test_tls_server_config(),
                    tunnel_config,
                    connection_config: ConnectionConfig::default(),
                    accept_timeout: Duration::from_secs(10),
                },
            )
            .await
        }
    });

    let client_handle = tokio::spawn(async move {
        connect_udp(UdpTransportConfig::new(server_addr, "localhost".into(), tunnel_config)).await
    });

    let (server_result, client_result) = tokio::join!(server_handle, client_handle);

    let server_transport = server_result.expect("server join").expect("server accept_udp");
    let client_transport = client_result.expect("client join").expect("client connect_udp");

    (client_transport, server_transport)
}

/// Verify the full connection sequence completes on loopback.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_stack_loopback_handshake() {
    let (client, server) = establish_loopback_pair().await;
    assert!(client.is_alive());
    assert!(server.is_alive());
    client.shutdown().await.expect("client shutdown");
    server.shutdown().await.expect("server shutdown");
}

/// Verify bidirectional data transfer through the tunnel.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_stack_bidirectional_data() {
    let (mut client, mut server) = establish_loopback_pair().await;

    // Client -> Server
    client.send(vec![0x01, 0x02, 0x03]).await.expect("client send");
    let received = server.recv().await.expect("server recv");
    assert_eq!(received, vec![0x01, 0x02, 0x03]);

    // Server -> Client
    server.send(vec![0x04, 0x05, 0x06]).await.expect("server send");
    let received = client.recv().await.expect("client recv");
    assert_eq!(received, vec![0x04, 0x05, 0x06]);

    client.shutdown().await.expect("client shutdown");
    server.shutdown().await.expect("server shutdown");
}

/// Verify multiple messages maintain ordering through the tunnel.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_stack_message_ordering() {
    let (client, mut server) = establish_loopback_pair().await;

    // Send multiple messages rapidly
    for i in 0u8..10 {
        client.send(vec![i]).await.expect("send");
    }

    // Verify they arrive in order
    for i in 0u8..10 {
        let received = server.recv().await.expect("recv");
        assert_eq!(received, vec![i], "message {i} arrived out of order");
    }

    client.shutdown().await.expect("shutdown");
    server.shutdown().await.expect("shutdown");
}

/// Verify that a larger payload survives the full stack.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_stack_larger_payload() {
    let (client, mut server) = establish_loopback_pair().await;

    // ~1 KB payload -- within a single RDPEUDP2 segment
    #[expect(clippy::cast_possible_truncation, reason = "i % 256 fits in u8")]
    let payload: Vec<u8> = (0..1000).map(|i: usize| (i % 256) as u8).collect();
    client.send(payload.clone()).await.expect("send");

    let received = server.recv().await.expect("recv");
    assert_eq!(received, payload);

    client.shutdown().await.expect("shutdown");
    server.shutdown().await.expect("shutdown");
}

/// Verify tunnel rejection when the server has a different security cookie.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tunnel_rejection_mismatched_cookie() {
    let server_sock = UdpSocket::bind("127.0.0.1:0").await.expect("bind");
    let server_addr = server_sock.local_addr().expect("addr");

    // Server expects a different cookie than the client will send
    let server_config = TunnelConfig {
        request_id: 42,
        security_cookie: [0xFF; 16],
    };
    let client_config = TunnelConfig {
        request_id: 42,
        security_cookie: [0xAA; 16],
    };

    let server_handle = tokio::spawn(async move {
        accept_udp(
            server_sock,
            UdpAcceptConfig {
                tls_config: test_tls_server_config(),
                tunnel_config: server_config,
                connection_config: ConnectionConfig::default(),
                accept_timeout: Duration::from_secs(10),
            },
        )
        .await
    });

    let client_handle = tokio::spawn(async move {
        connect_udp(UdpTransportConfig::new(server_addr, "localhost".into(), client_config)).await
    });

    let (server_result, client_result) = tokio::join!(server_handle, client_handle);

    // At least one side should fail -- the server rejects the mismatched cookie
    // and sends a failure CreateResponse, which the client receives.
    let server_err = server_result.expect("server join").is_err();
    let client_err = client_result.expect("client join").is_err();

    assert!(
        server_err || client_err,
        "at least one side should fail with mismatched cookies"
    );
}
