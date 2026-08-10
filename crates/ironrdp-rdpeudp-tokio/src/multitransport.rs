//! Multitransport bootstrapping orchestrator.
//!
//! When an RDP server sends an Initiate Multitransport Request PDU on
//! the TCP connection, the client must:
//!
//! 1. Parse the request (requestId + requestedProtocol + securityCookie)
//! 2. Establish a UDP transport using `connect_udp()`
//! 3. Send an Initiate Multitransport Response PDU (S_OK or E_ABORT)
//!    back on the TCP connection
//!
//! `MultitransportBootstrap` orchestrates this sequence. The upstream
//! `ironrdp-connector` doesn't implement multitransport yet, so this
//! acts as a standalone shim that an application wires into its
//! connection flow.

use core::net::SocketAddr;

use ironrdp_rdpemt::{MultitransportRequest, MultitransportResponse, RdpemtError, RdpemtErrorExt as _};
use ironrdp_rdpeudp::ConnectionConfig;

use crate::error::{UdpTransportError, UdpTransportErrorExt as _};
use crate::transport::{UdpTransport, UdpTransportConfig, connect_udp};

/// Orchestrates the multitransport connection sequence.
///
/// Created from the raw Initiate Multitransport Request PDU payload
/// received on the TCP MCS message channel. Call [`connect()`] to
/// establish the UDP transport, then [`response_pdu()`] to get the
/// bytes to send back on TCP.
///
/// [`connect()`]: MultitransportBootstrap::connect
/// [`response_pdu()`]: MultitransportBootstrap::response_pdu
///
/// # Example
///
/// ```ignore
/// // In the TCP connection handler, after receiving MultitransportRequest:
/// let mut bootstrap = MultitransportBootstrap::new(request);
///
/// // Attempt the UDP connection
/// let _ = bootstrap.connect(server_addr, "server.example.com".into(), Default::default()).await;
///
/// // Always send the response back on TCP (S_OK or E_ABORT)
/// let response_bytes = bootstrap.response_pdu().expect("response available after connect");
/// tcp_writer.write_all(&response_bytes).await?;
///
/// // If successful, use the transport
/// if let Some(transport) = bootstrap.take_transport() {
///     // ... use transport for DVC data
/// }
/// ```
pub struct MultitransportBootstrap {
    request: MultitransportRequest,
    transport: Option<UdpTransport>,
    response: Option<MultitransportResponse>,
}

impl MultitransportBootstrap {
    /// Create from a parsed `MultitransportRequest`.
    pub fn new(request: MultitransportRequest) -> Self {
        Self {
            request,
            transport: None,
            response: None,
        }
    }

    /// Parse from raw PDU payload bytes.
    ///
    /// The payload is the body of the Initiate Multitransport Request
    /// PDU (after TPKT + X224 + MCS + security header have been
    /// stripped by the existing ironrdp-pdu stack).
    pub fn from_pdu(pdu_payload: &[u8]) -> Result<Self, UdpTransportError> {
        let request: MultitransportRequest = ironrdp_core::decode(pdu_payload)
            .map_err(|error| UdpTransportError::rdpemt("decode multitransport request", RdpemtError::decode(error)))?;

        Ok(Self::new(request))
    }

    /// Attempt to establish the UDP transport.
    ///
    /// On success, stores the transport and prepares an `S_OK` response.
    /// On failure, prepares an `E_ABORT` response and returns the error.
    ///
    /// After calling this, use [`response_pdu()`] to get the bytes to
    /// send back to the server on the TCP connection.
    ///
    /// [`response_pdu()`]: MultitransportBootstrap::response_pdu
    pub async fn connect(
        &mut self,
        server_addr: SocketAddr,
        server_name: String,
        connection_config: ConnectionConfig,
    ) -> Result<(), UdpTransportError> {
        let tunnel_config = self.request.to_tunnel_config();
        let mut config = UdpTransportConfig::new(server_addr, server_name, tunnel_config);
        config.connection_config = connection_config;

        match connect_udp(config).await {
            Ok(transport) => {
                self.transport = Some(transport);
                self.response = Some(MultitransportResponse::success(self.request.request_id));
                Ok(())
            }
            Err(e) => {
                self.response = Some(MultitransportResponse::abort(self.request.request_id));
                Err(e)
            }
        }
    }

    /// Get the response PDU bytes to send back on the TCP connection.
    ///
    /// Returns `None` if [`connect()`] hasn't been called yet.
    ///
    /// # Panics
    ///
    /// Panics if the response fails to encode. The response is a fixed-size
    /// structure built by this crate, so a failure here means the encoder and
    /// the type have gone out of sync rather than anything a caller can cause.
    ///
    /// [`connect()`]: MultitransportBootstrap::connect
    pub fn response_pdu(&self) -> Option<Vec<u8>> {
        self.response
            .as_ref()
            .map(|r| ironrdp_core::encode_vec(r).expect("MultitransportResponse encoding cannot fail"))
    }

    /// The original request from the server.
    pub fn request(&self) -> &MultitransportRequest {
        &self.request
    }

    /// Take ownership of the established UDP transport.
    ///
    /// Returns `None` if the connection failed or hasn't been attempted.
    pub fn take_transport(&mut self) -> Option<UdpTransport> {
        self.transport.take()
    }

    /// Whether the UDP transport was established.
    pub fn is_connected(&self) -> bool {
        self.transport.is_some()
    }
}

#[cfg(test)]
mod tests {
    use ironrdp_rdpemt::TunnelConfig;

    use super::*;

    #[test]
    fn new_from_request() {
        let request = MultitransportRequest {
            request_id: 42,
            requested_protocol: ironrdp_rdpemt::RequestedProtocol::ReliableUdp,
            security_cookie: [0xAB; 16],
        };

        let bootstrap = MultitransportBootstrap::new(request.clone());
        assert_eq!(bootstrap.request().request_id, 42);
        assert!(!bootstrap.is_connected());
        assert!(bootstrap.response_pdu().is_none());
    }

    #[test]
    fn from_pdu_roundtrip() {
        let request = MultitransportRequest {
            request_id: 99,
            requested_protocol: ironrdp_rdpemt::RequestedProtocol::ReliableUdp,
            security_cookie: [0xCC; 16],
        };

        let encoded = ironrdp_core::encode_vec(&request).expect("encode");
        let bootstrap = MultitransportBootstrap::from_pdu(&encoded).expect("decode");
        assert_eq!(bootstrap.request().request_id, 99);
        assert_eq!(bootstrap.request().security_cookie, [0xCC; 16]);
    }

    #[test]
    fn from_pdu_rejects_garbage() {
        let result = MultitransportBootstrap::from_pdu(&[0xFF, 0xFF]);
        assert!(result.is_err());
    }

    #[test]
    fn tunnel_config_extraction() {
        let request = MultitransportRequest {
            request_id: 7,
            requested_protocol: ironrdp_rdpemt::RequestedProtocol::ReliableUdp,
            security_cookie: [0xDD; 16],
        };

        let config: TunnelConfig = request.to_tunnel_config();
        assert_eq!(config.request_id, 7);
        assert_eq!(config.security_cookie, [0xDD; 16]);
    }
}
