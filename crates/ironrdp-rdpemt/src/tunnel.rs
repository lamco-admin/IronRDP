//! RDPEMT tunnel state machine.
//!
//! Sans-I/O: operates on decrypted PDU bytes (after TLS) and produces
//! plaintext PDU bytes (for TLS encryption). Never touches the network.
//!
//! # Lifecycle
//!
//! ```text
//! Client: Created ──(poll_pdu → CreateRequest)──→ AwaitingResponse
//!         AwaitingResponse ──(CreateResponse S_OK)──→ Established
//!         AwaitingResponse ──(CreateResponse !S_OK)──→ Failed
//!
//! Server: Created ──(CreateRequest, cookie match)──→ Established
//!         Created ──(CreateRequest, no match)──→ Failed
//!
//! Either: Established ──(send_data / handle Data PDU)──→ bidirectional data
//! ```

use alloc::collections::VecDeque;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::error::{RdpemtError, RdpemtErrorExt as _};
use crate::pdu::create_request::SECURITY_COOKIE_LEN;
use crate::pdu::{TunnelCreateRequest, TunnelCreateResponse, TunnelData, TunnelPdu};

// ════════════════════════════════════════════════════════════════════
// Public types
// ════════════════════════════════════════════════════════════════════

/// Which role this tunnel endpoint plays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Initiated the tunnel (sends CreateRequest).
    Client,
    /// Accepts the tunnel (validates CreateRequest, sends CreateResponse).
    Server,
}

/// Events produced by the tunnel for the application layer.
///
/// Retrieved by calling `poll_event()` after `handle_pdu()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TunnelEvent {
    /// The tunnel has been established (CreateResponse with S_OK).
    Established,
    /// Higher-layer data received through the tunnel.
    Data(Vec<u8>),
    /// The tunnel creation failed (non-recoverable).
    Failed {
        /// The HRESULT from the CreateResponse, or 0xFFFFFFFF for server-side
        /// cookie mismatch.
        hr_response: u32,
    },
}

/// Configuration for creating a tunnel.
///
/// These values come from the Initiate Multitransport Request PDU
/// that the server sends over the main TCP connection
/// (MS-RDPBCGR Section 2.2.15.1).
#[derive(Debug, Clone)]
pub struct TunnelConfig {
    /// The request ID from the Initiate Multitransport Request PDU.
    pub request_id: u32,
    /// The 16-byte security cookie from the Initiate Multitransport Request PDU.
    pub security_cookie: [u8; SECURITY_COOKIE_LEN],
}

// ════════════════════════════════════════════════════════════════════
// Internal state
// ════════════════════════════════════════════════════════════════════

/// Tunnel lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TunnelState {
    /// Initial state. Client will produce CreateRequest; server awaits it.
    Created,
    /// Client has sent CreateRequest, awaiting CreateResponse.
    AwaitingResponse,
    /// Tunnel is active: Data PDUs can flow both directions.
    Established,
    /// Tunnel creation failed (non-recoverable).
    Failed,
}

// ════════════════════════════════════════════════════════════════════
// RdpemtTunnel
// ════════════════════════════════════════════════════════════════════

/// RDPEMT tunnel state machine.
///
/// Sans-I/O: operates on decrypted PDU bytes (post-TLS) and produces
/// plaintext PDU bytes (pre-TLS). The async driver handles
/// TLS encryption/decryption and RDPEUDP2 transport.
///
/// # Client usage
///
/// ```ignore
/// let mut tunnel = RdpemtTunnel::client(config);
///
/// // 1. Send the CreateRequest
/// let create_req = tunnel.poll_pdu().expect("CreateRequest");
/// tls_send(&create_req);
///
/// // 2. Receive CreateResponse
/// let response_bytes = tls_recv();
/// tunnel.handle_pdu(&response_bytes)?;
/// while let Some(event) = tunnel.poll_event() {
///     match event {
///         TunnelEvent::Established => { /* ready */ }
///         TunnelEvent::Failed { hr_response } => { /* disconnect */ }
///         _ => {}
///     }
/// }
///
/// // 3. Exchange data
/// tunnel.send_data(b"hello")?;
/// let data_pdu = tunnel.poll_pdu().expect("Data PDU");
/// tls_send(&data_pdu);
/// ```
pub struct RdpemtTunnel {
    side: Side,
    state: TunnelState,
    config: TunnelConfig,
    /// PDUs waiting to be sent (encoded wire bytes).
    outgoing: VecDeque<Vec<u8>>,
    /// Events waiting to be delivered to the application.
    events: VecDeque<TunnelEvent>,
}

impl RdpemtTunnel {
    /// Create a client-side tunnel.
    ///
    /// Immediately enqueues the CreateRequest PDU. Call `poll_pdu()` to
    /// retrieve it, then send through TLS.
    pub fn client(config: TunnelConfig) -> Self {
        let mut tunnel = Self {
            side: Side::Client,
            state: TunnelState::Created,
            config,
            outgoing: VecDeque::new(),
            events: VecDeque::new(),
        };
        tunnel.enqueue_create_request();
        tunnel
    }

    /// Create a server-side tunnel.
    ///
    /// The tunnel waits for a CreateRequest from the client. The config
    /// provides the expected request_id + security_cookie for validation.
    pub fn server(config: TunnelConfig) -> Self {
        Self {
            side: Side::Server,
            state: TunnelState::Created,
            config,
            outgoing: VecDeque::new(),
            events: VecDeque::new(),
        }
    }

    /// Process a received (already TLS-decrypted) PDU.
    ///
    /// After calling this, use `poll_event()` to retrieve any events
    /// and `poll_pdu()` to retrieve any response PDUs.
    pub fn handle_pdu(&mut self, data: &[u8]) -> Result<(), RdpemtError> {
        let pdu: TunnelPdu = ironrdp_core::decode(data).map_err(RdpemtError::decode)?;

        match (&self.state, &self.side, pdu) {
            // Client in AwaitingResponse receives CreateResponse
            (TunnelState::AwaitingResponse, Side::Client, TunnelPdu::CreateResponse(resp)) => {
                if resp.is_success() {
                    self.state = TunnelState::Established;
                    self.events.push_back(TunnelEvent::Established);
                } else {
                    self.state = TunnelState::Failed;
                    self.events.push_back(TunnelEvent::Failed {
                        hr_response: resp.hr_response,
                    });
                }
            }

            // Server in Created receives CreateRequest
            (TunnelState::Created, Side::Server, TunnelPdu::CreateRequest(req)) => {
                if req.request_id == self.config.request_id && req.security_cookie == self.config.security_cookie {
                    self.state = TunnelState::Established;
                    self.enqueue_create_response(TunnelCreateResponse::S_OK);
                    self.events.push_back(TunnelEvent::Established);
                } else {
                    self.state = TunnelState::Failed;
                    // E_ACCESSDENIED
                    let hr = 0x80070005u32;
                    self.enqueue_create_response(hr);
                    self.events.push_back(TunnelEvent::Failed { hr_response: hr });
                }
            }

            // Either side in Established receives Data
            (TunnelState::Established, _, TunnelPdu::Data(data_pdu)) => {
                self.events.push_back(TunnelEvent::Data(data_pdu.higher_layer_data));
            }

            // Any other combination is a state violation
            _ => return Err(RdpemtError::invalid_state("handle tunnel PDU")),
        }

        Ok(())
    }

    /// Retrieve the next outgoing PDU to send (encoded, needs TLS encryption).
    ///
    /// Returns `None` when no PDU is pending. Call in a loop until `None`.
    pub fn poll_pdu(&mut self) -> Option<Vec<u8>> {
        self.outgoing.pop_front()
    }

    /// Retrieve the next event for the application layer.
    ///
    /// Returns `None` when no events are pending. Call in a loop until `None`.
    pub fn poll_event(&mut self) -> Option<TunnelEvent> {
        self.events.pop_front()
    }

    /// Queue application data for transmission as a Data PDU.
    ///
    /// The data is wrapped in a TunnelData PDU and enqueued for `poll_pdu()`.
    /// Returns an error if the tunnel is not in the Established state.
    pub fn send_data(&mut self, data: &[u8]) -> Result<(), RdpemtError> {
        if self.state != TunnelState::Established {
            return Err(RdpemtError::invalid_state("send tunnel data"));
        }

        let pdu = TunnelData {
            sub_headers: Vec::new(),
            higher_layer_data: data.to_vec(),
        };

        let encoded = ironrdp_core::encode_vec(&pdu).map_err(RdpemtError::encode)?;
        self.outgoing.push_back(encoded);

        Ok(())
    }

    /// Whether the tunnel is established and ready for data transfer.
    pub fn is_established(&self) -> bool {
        self.state == TunnelState::Established
    }

    /// Whether the tunnel creation has failed.
    pub fn is_failed(&self) -> bool {
        self.state == TunnelState::Failed
    }

    /// Which role this endpoint plays.
    pub fn side(&self) -> Side {
        self.side
    }

    // ── Internal helpers ──

    fn enqueue_create_request(&mut self) {
        let pdu = TunnelCreateRequest {
            request_id: self.config.request_id,
            security_cookie: self.config.security_cookie,
        };

        let encoded = ironrdp_core::encode_vec(&pdu).expect("CreateRequest encoding is infallible for valid inputs");
        self.outgoing.push_back(encoded);
        self.state = TunnelState::AwaitingResponse;
    }

    fn enqueue_create_response(&mut self, hr_response: u32) {
        let pdu = TunnelCreateResponse { hr_response };

        let encoded = ironrdp_core::encode_vec(&pdu).expect("CreateResponse encoding is infallible for valid inputs");
        self.outgoing.push_back(encoded);
    }
}

impl core::fmt::Debug for RdpemtTunnel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RdpemtTunnel")
            .field("side", &self.side)
            .field("state", &self.state)
            .field("outgoing_count", &self.outgoing.len())
            .field("events_count", &self.events.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> TunnelConfig {
        TunnelConfig {
            request_id: 7,
            security_cookie: [
                0xe2, 0xf0, 0xd1, 0x08, 0x56, 0x7f, 0xb4, 0x3a, 0xdc, 0xf4, 0xb3, 0xdc, 0x16, 0x92, 0x1e, 0x3a,
            ],
        }
    }

    // ── Client happy path ──

    #[test]
    fn client_produces_create_request_on_construction() {
        let tunnel = RdpemtTunnel::client(test_config());
        assert!(!tunnel.is_established());
        assert!(!tunnel.is_failed());
        assert_eq!(tunnel.side(), Side::Client);
    }

    #[test]
    fn client_poll_pdu_returns_create_request() {
        let mut tunnel = RdpemtTunnel::client(test_config());
        let pdu_bytes = tunnel.poll_pdu().expect("should have CreateRequest");

        // Verify it decodes as a CreateRequest
        let pdu: TunnelPdu = ironrdp_core::decode(&pdu_bytes).expect("decode");
        match pdu {
            TunnelPdu::CreateRequest(req) => {
                assert_eq!(req.request_id, 7);
                assert_eq!(req.security_cookie, test_config().security_cookie);
            }
            other => panic!("expected CreateRequest, got {other:?}"),
        }

        // No more PDUs
        assert!(tunnel.poll_pdu().is_none());
    }

    #[test]
    fn client_handles_successful_create_response() {
        let mut tunnel = RdpemtTunnel::client(test_config());
        let _create_req = tunnel.poll_pdu().expect("CreateRequest");

        // Build a successful CreateResponse
        let response = TunnelCreateResponse {
            hr_response: TunnelCreateResponse::S_OK,
        };
        let response_bytes = ironrdp_core::encode_vec(&response).expect("encode");

        tunnel.handle_pdu(&response_bytes).expect("handle ok");
        assert!(tunnel.is_established());

        let event = tunnel.poll_event().expect("should have Established event");
        assert_eq!(event, TunnelEvent::Established);
        assert!(tunnel.poll_event().is_none());
    }

    #[test]
    fn client_handles_rejected_create_response() {
        let mut tunnel = RdpemtTunnel::client(test_config());
        let _create_req = tunnel.poll_pdu().expect("CreateRequest");

        let response = TunnelCreateResponse {
            hr_response: 0x8000_0001,
        };
        let response_bytes = ironrdp_core::encode_vec(&response).expect("encode");

        tunnel.handle_pdu(&response_bytes).expect("handle ok");
        assert!(tunnel.is_failed());
        assert!(!tunnel.is_established());

        let event = tunnel.poll_event().expect("should have Failed event");
        assert_eq!(
            event,
            TunnelEvent::Failed {
                hr_response: 0x8000_0001
            }
        );
    }

    // ── Server happy path ──

    #[test]
    fn server_starts_in_created() {
        let tunnel = RdpemtTunnel::server(test_config());
        assert!(!tunnel.is_established());
        assert!(!tunnel.is_failed());
        assert_eq!(tunnel.side(), Side::Server);
        // No outgoing PDUs initially
    }

    #[test]
    fn server_handles_matching_create_request() {
        let mut server = RdpemtTunnel::server(test_config());

        let request = TunnelCreateRequest {
            request_id: 7,
            security_cookie: test_config().security_cookie,
        };
        let request_bytes = ironrdp_core::encode_vec(&request).expect("encode");

        server.handle_pdu(&request_bytes).expect("handle ok");
        assert!(server.is_established());

        // Server should have enqueued a successful CreateResponse
        let response_bytes = server.poll_pdu().expect("should have CreateResponse");
        let pdu: TunnelPdu = ironrdp_core::decode(&response_bytes).expect("decode");
        match pdu {
            TunnelPdu::CreateResponse(resp) => {
                assert!(resp.is_success());
            }
            other => panic!("expected CreateResponse, got {other:?}"),
        }

        let event = server.poll_event().expect("should have Established event");
        assert_eq!(event, TunnelEvent::Established);
    }

    #[test]
    fn server_rejects_mismatching_request_id() {
        let mut server = RdpemtTunnel::server(test_config());

        let request = TunnelCreateRequest {
            request_id: 999, // wrong ID
            security_cookie: test_config().security_cookie,
        };
        let request_bytes = ironrdp_core::encode_vec(&request).expect("encode");

        server.handle_pdu(&request_bytes).expect("handle ok");
        assert!(server.is_failed());

        // Server sends a rejection response
        let response_bytes = server.poll_pdu().expect("should have CreateResponse");
        let pdu: TunnelPdu = ironrdp_core::decode(&response_bytes).expect("decode");
        match pdu {
            TunnelPdu::CreateResponse(resp) => {
                assert!(!resp.is_success());
            }
            other => panic!("expected CreateResponse, got {other:?}"),
        }

        let event = server.poll_event().expect("should have Failed event");
        match event {
            TunnelEvent::Failed { .. } => {}
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn server_rejects_mismatching_cookie() {
        let mut server = RdpemtTunnel::server(test_config());

        let request = TunnelCreateRequest {
            request_id: 7,
            security_cookie: [0xFF; 16], // wrong cookie
        };
        let request_bytes = ironrdp_core::encode_vec(&request).expect("encode");

        server.handle_pdu(&request_bytes).expect("handle ok");
        assert!(server.is_failed());
    }

    // ── Data transfer ──

    #[test]
    fn established_tunnel_sends_data() {
        let mut tunnel = RdpemtTunnel::client(test_config());
        let _create_req = tunnel.poll_pdu();

        let response = TunnelCreateResponse {
            hr_response: TunnelCreateResponse::S_OK,
        };
        let response_bytes = ironrdp_core::encode_vec(&response).expect("encode");
        tunnel.handle_pdu(&response_bytes).expect("handle ok");
        let _event = tunnel.poll_event();

        // Send data
        tunnel.send_data(b"hello world").expect("send ok");
        let data_pdu_bytes = tunnel.poll_pdu().expect("should have Data PDU");

        let pdu: TunnelPdu = ironrdp_core::decode(&data_pdu_bytes).expect("decode");
        match pdu {
            TunnelPdu::Data(data) => {
                assert_eq!(data.higher_layer_data, b"hello world");
            }
            other => panic!("expected Data, got {other:?}"),
        }
    }

    #[test]
    fn established_tunnel_receives_data() {
        let mut tunnel = RdpemtTunnel::client(test_config());
        let _create_req = tunnel.poll_pdu();

        let response = TunnelCreateResponse {
            hr_response: TunnelCreateResponse::S_OK,
        };
        let response_bytes = ironrdp_core::encode_vec(&response).expect("encode");
        tunnel.handle_pdu(&response_bytes).expect("handle ok");
        let _event = tunnel.poll_event();

        // Simulate receiving a Data PDU
        let incoming = TunnelData {
            sub_headers: Vec::new(),
            higher_layer_data: b"from server".to_vec(),
        };
        let incoming_bytes = ironrdp_core::encode_vec(&incoming).expect("encode");

        tunnel.handle_pdu(&incoming_bytes).expect("handle ok");

        let event = tunnel.poll_event().expect("should have Data event");
        assert_eq!(event, TunnelEvent::Data(b"from server".to_vec()));
    }

    // ── State violation tests ──

    #[test]
    fn send_data_before_established_fails() {
        let mut tunnel = RdpemtTunnel::client(test_config());
        let _create_req = tunnel.poll_pdu();

        let result = tunnel.send_data(b"too early");
        assert!(result.is_err());
    }

    #[test]
    fn handle_data_before_established_fails() {
        let mut tunnel = RdpemtTunnel::client(test_config());
        let _create_req = tunnel.poll_pdu();

        let data = TunnelData {
            sub_headers: Vec::new(),
            higher_layer_data: b"premature".to_vec(),
        };
        let data_bytes = ironrdp_core::encode_vec(&data).expect("encode");

        let result = tunnel.handle_pdu(&data_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn handle_create_request_on_established_client_fails() {
        let mut tunnel = RdpemtTunnel::client(test_config());
        let _create_req = tunnel.poll_pdu();

        let response = TunnelCreateResponse {
            hr_response: TunnelCreateResponse::S_OK,
        };
        let response_bytes = ironrdp_core::encode_vec(&response).expect("encode");
        tunnel.handle_pdu(&response_bytes).expect("handle ok");
        let _event = tunnel.poll_event();

        // Try to handle a CreateRequest on an established client: invalid
        let request = TunnelCreateRequest {
            request_id: 7,
            security_cookie: test_config().security_cookie,
        };
        let request_bytes = ironrdp_core::encode_vec(&request).expect("encode");

        let result = tunnel.handle_pdu(&request_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn send_data_on_failed_tunnel_fails() {
        let mut tunnel = RdpemtTunnel::client(test_config());
        let _create_req = tunnel.poll_pdu();

        let response = TunnelCreateResponse {
            hr_response: 0x8000_FFFF,
        };
        let response_bytes = ironrdp_core::encode_vec(&response).expect("encode");
        tunnel.handle_pdu(&response_bytes).expect("handle ok");

        let result = tunnel.send_data(b"dead tunnel");
        assert!(result.is_err());
    }

    // ── End-to-end: client + server ──

    #[test]
    fn end_to_end_handshake_and_data() {
        let config = test_config();

        // Create both sides
        let mut client = RdpemtTunnel::client(config.clone());
        let mut server = RdpemtTunnel::server(config);

        // Client produces CreateRequest
        let create_req_bytes = client.poll_pdu().expect("client CreateRequest");
        assert!(client.poll_pdu().is_none());

        // Server processes CreateRequest, produces CreateResponse
        server
            .handle_pdu(&create_req_bytes)
            .expect("server handle CreateRequest");
        assert!(server.is_established());
        let create_resp_bytes = server.poll_pdu().expect("server CreateResponse");
        assert!(server.poll_pdu().is_none());

        // Client processes CreateResponse
        client
            .handle_pdu(&create_resp_bytes)
            .expect("client handle CreateResponse");
        assert!(client.is_established());

        // Drain events
        assert_eq!(server.poll_event(), Some(TunnelEvent::Established));
        assert_eq!(client.poll_event(), Some(TunnelEvent::Established));

        // Client sends data to server
        client.send_data(b"request data").expect("client send");
        let data_to_server = client.poll_pdu().expect("client Data PDU");

        server.handle_pdu(&data_to_server).expect("server handle Data");
        assert_eq!(server.poll_event(), Some(TunnelEvent::Data(b"request data".to_vec())));

        // Server sends data to client
        server.send_data(b"response data").expect("server send");
        let data_to_client = server.poll_pdu().expect("server Data PDU");

        client.handle_pdu(&data_to_client).expect("client handle Data");
        assert_eq!(client.poll_event(), Some(TunnelEvent::Data(b"response data".to_vec())));
    }
}
