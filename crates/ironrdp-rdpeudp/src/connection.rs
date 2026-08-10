//! RDPEUDP2 connection state machine.
//!
//! Sans-I/O design: the state machine never touches the network or system
//! clock. All `MonotonicInstant` values are passed in by the caller, and outgoing
//! packets are returned as `Transmit` values. This mirrors Quinn's
//! `Connection` architecture (ADR-0001).
//!
//! # Lifecycle
//!
//! ```text
//! Client: connect() → SynSent ──(recv SYN+ACK)──→ Established
//! Server: accept()  → SynReceived ──(recv ACK)──→ Established
//! Either: close()   → Closed
//! Either: idle timeout ──→ Closed
//! ```
//!
//! # V1/V2 packet discrimination
//!
//! The v1 wire format is used for the three-way handshake (SYN/SYN+ACK/ACK).
//! Once established, all packets use the v2 format with the PacketPrefixByte
//! framing from MS-RDPEUDP2 Section 2.2.1.3.

use crate::time::MonotonicInstant;
use alloc::collections::VecDeque;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::time::Duration;

use ironrdp_core::{decode, encode_vec};

use crate::congestion::CongestionControl;
use crate::error::{RdpeudpError, RdpeudpErrorExt as _};
use crate::loss::LossDetector;
use crate::pdu::prefix::{decode_with_prefix, encode_with_prefix};
use crate::pdu::v1_ack::{V1AckVectorElement, V1AckVectorHeader, VectorElementState};
use crate::pdu::v1_flags::V1Flags;
use crate::pdu::v1_header::FecHeader;
use crate::pdu::v1_syn::{SynDataExPayload, SynDataPayload, SynExFlags, UdpVersion};
use crate::pdu::v2_ack::{AckPayload, AckVectorEntry, AckVectorPayload};
use crate::pdu::v2_control::AckOfAcksPayload;
use crate::pdu::v2_data::{DataBody, DataHeader};
use crate::pdu::v2_flags::V2Flags;
use crate::pdu::v2_header::V2Header;
use crate::pdu::{V1Datagram, V2Packet};
use crate::recv_window::RecvWindow;
use crate::reliability::ReliabilityController;
use crate::rtt::RttEstimator;
use crate::send_window::SendWindow;
use crate::seq;
use crate::timer::{Timer, TimerTable};

// ════════════════════════════════════════════════════════════════════
// Public types
// ════════════════════════════════════════════════════════════════════

/// Which side of the connection this endpoint represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Initiated the connection (sent the initial SYN).
    Client,
    /// Accepted the connection (responded with SYN+ACK).
    Server,
}

/// Events produced by the state machine for the application layer.
///
/// Retrieved by calling `poll_event()` after any state mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// The handshake completed and the connection is established.
    Connected,

    /// Application data has been received and reassembled in order.
    DataReceived(Vec<u8>),

    /// The connection has been closed (locally or by timeout).
    ConnectionClosed,
}

/// An outgoing packet to be sent on the wire.
///
/// Retrieved by calling `poll_transmit()` after any state mutation.
#[derive(Debug, Clone)]
pub struct Transmit {
    /// The raw wire bytes to send (including PacketPrefixByte for v2 packets).
    pub contents: Vec<u8>,
}

/// Configuration for a new RDPEUDP2 connection.
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    /// Initial sequence number from the SYN exchange.
    pub initial_sequence_number: u32,

    /// log2 of the receive window size (0..=15).
    /// Actual window = `1 << log_window_size`.
    pub log_window_size: u8,

    /// Upstream MTU (client → server), negotiated during handshake.
    pub upstream_mtu: u16,

    /// Downstream MTU (server → client), negotiated during handshake.
    pub downstream_mtu: u16,

    /// Idle timeout: connection closes if no packets received within this duration.
    ///
    /// Default: 65 seconds, the interval [MS-RDPEUDP] 3.1.1.9 gives for
    /// deciding the peer has gone. Shortening it below that closes
    /// connections a conforming peer still considers live.
    pub idle_timeout: Duration,

    /// Keep-alive interval: send a probe if no data has been sent for this long.
    /// Should be shorter than the remote's idle timeout.
    ///
    /// Default: 8 seconds. The spec asks only that endpoints acknowledge
    /// "periodically" (3.1.1.9), the point being to hold the NAT binding
    /// open, so the interval is ours to pick.
    pub keep_alive_interval: Duration,

    /// ACK delay timeout: maximum time to hold an ACK before sending it.
    /// Default: 50ms per MS-RDPEUDP2.
    pub ack_delay_timeout: Duration,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            initial_sequence_number: 0,
            log_window_size: 6,
            upstream_mtu: 1232,
            downstream_mtu: 1232,
            idle_timeout: Duration::from_secs(65),
            keep_alive_interval: Duration::from_secs(8),
            ack_delay_timeout: Duration::from_millis(50),
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Internal state
// ════════════════════════════════════════════════════════════════════

/// How many unanswered retransmits of a handshake datagram we tolerate
/// before closing.
///
/// [MS-RDPEUDP] 3.1.5.4.1 puts the cutoff at "at least three and no more
/// than five", so an endpoint may not give up earlier than three and must
/// have given up by five. Five is the friendliest conforming choice on a
/// link that is losing packets.
const HANDSHAKE_RETRANSMIT_LIMIT: u8 = 5;

/// Connection state in the handshake / lifecycle FSM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Client has sent SYN, waiting for SYN+ACK.
    SynSent,
    /// Server has received SYN and sent SYN+ACK, waiting for final ACK.
    SynReceived,
    /// Handshake complete, data transfer active.
    Established,
    /// Connection terminated.
    Closed,
}

/// An acknowledgment ready to go onto an outgoing packet.
///
/// The cumulative and vector forms are mutually exclusive: [MS-RDPEUDP2]
/// 2.2.1.1 says the ACKVEC flag "MUST NOT be set if the ACK flag is set".
struct Acknowledgement {
    flag: V2Flags,
    ack: Option<AckPayload>,
    ack_vector: Option<AckVectorPayload>,
}

/// Negotiated parameters from the handshake.
#[derive(Debug, Clone)]
struct NegotiatedParams {
    /// Our ISN (from our SYN).
    local_isn: u32,
    /// Remote ISN (from their SYN).
    remote_isn: u32,
    /// Negotiated MTU: min(upstream, downstream, remote_upstream, remote_downstream).
    mtu: u16,
    /// log2 of the window size.
    log_window_size: u8,
}

// ════════════════════════════════════════════════════════════════════
// Connection
// ════════════════════════════════════════════════════════════════════

/// RDPEUDP2 connection state machine.
///
/// Wires together all protocol components (send/receive windows,
/// congestion control, loss detection, RTT estimation, timers,
/// reliability controller) behind a sans-I/O polling API.
///
/// # Usage pattern
///
/// ```ignore
/// // Create
/// let mut conn = RdpeudpConnection::connect(config, now);
///
/// // Main loop:
/// // 1. Send the initial SYN
/// while let Some(transmit) = conn.poll_transmit() {
///     socket.send(&transmit.contents);
/// }
///
/// // 2. Receive packets
/// conn.handle_datagram(&mut incoming_bytes, now)?;
///
/// // 3. Check for events
/// while let Some(event) = conn.poll_event() {
///     match event {
///         Event::Connected => { /* handshake done */ }
///         Event::DataReceived(data) => { /* process data */ }
///         Event::ConnectionClosed => { break; }
///     }
/// }
///
/// // 4. Handle timeouts
/// if let Some(deadline) = conn.poll_timeout() {
///     // schedule wakeup at `deadline`
/// }
/// conn.handle_timeout(now);
/// ```
pub struct RdpeudpConnection {
    /// Which side we are.
    side: Side,

    /// Current lifecycle state.
    state: State,

    /// Configuration (immutable after construction).
    config: ConnectionConfig,

    /// Negotiated handshake parameters (populated during/after handshake).
    params: Option<NegotiatedParams>,

    // ── Data transfer components (initialized on Established) ──
    send_window: Option<SendWindow>,
    recv_window: Option<RecvWindow>,
    congestion: CongestionControl,
    loss_detector: LossDetector,
    reliability: ReliabilityController,
    rtt: RttEstimator,
    timers: TimerTable,

    /// Queued outgoing wire-format packets.
    pending_transmits: VecDeque<Transmit>,

    /// Queued events for the application.
    pending_events: VecDeque<Event>,

    /// Application data waiting to be packaged into packets.
    send_buffer: VecDeque<Vec<u8>>,

    /// Whether we need to send a standalone ACK (no data piggybacked).
    ack_pending: bool,

    /// Whether we need to set the CWR flag on the next data packet.
    cwr_pending: bool,

    /// Whether the remote has gaps (we should set CN on our ACKs).
    cn_pending: bool,

    /// The AckOfAcks sequence number to send (if any).
    pending_ack_of_acks: Option<u16>,

    /// The last handshake datagram we put on the wire, kept so it can go
    /// out again. [MS-RDPEUDP] 1.3.1 delivers the SYN, the SYN+ACK and the
    /// ACK by persistent retransmits whatever mode the transport is in.
    ///
    /// The client holds on to its final ACK after establishing, because a
    /// repeated SYN+ACK is the server saying it never arrived.
    handshake_datagram: Option<Vec<u8>>,

    /// How many times `handshake_datagram` has been retransmitted by the
    /// timer. Peer-prompted resends do not count: those got an answer.
    handshake_retransmits: u8,

    /// Timestamp reference for reconstructing 24-bit wire timestamps.
    /// Tracks our local timestamp counter in 4μs units.
    local_timestamp_ref: u64,

    /// Remote timestamp reference for reconstruction.
    remote_timestamp_ref: u64,

    /// Reusable wire encoding buffer.
    wire_buf: Vec<u8>,
}

impl RdpeudpConnection {
    // ════════════════════════════════════════════════════════════════
    // Constructors
    // ════════════════════════════════════════════════════════════════

    /// Create a client-side connection and enqueue the initial SYN.
    ///
    /// After calling this, use `poll_transmit()` to retrieve the SYN
    /// datagram to send on the wire.
    pub fn connect(config: ConnectionConfig, now: MonotonicInstant) -> Self {
        let mut conn = Self::new(Side::Client, config);
        conn.enqueue_syn(now);
        conn
    }

    /// Create a server-side connection from a received SYN datagram.
    ///
    /// The caller has already decoded the V1 datagram and determined
    /// it's a SYN. This constructs the server state machine and
    /// enqueues the SYN+ACK response.
    ///
    /// Returns an error if the SYN datagram is malformed (missing
    /// SYN data or version negotiation).
    pub fn accept(
        config: ConnectionConfig,
        syn_datagram: &V1Datagram,
        now: MonotonicInstant,
    ) -> Result<Self, RdpeudpError> {
        let syn_data = syn_datagram
            .syn_data
            .as_ref()
            .ok_or_else(|| RdpeudpError::invalid_packet("accept", "SYN datagram missing SynData payload"))?;

        let syn_data_ex = syn_datagram
            .syn_data_ex
            .as_ref()
            .ok_or_else(|| RdpeudpError::invalid_packet("accept", "SYN datagram missing SynDataEx payload"))?;

        // We only support v2+
        if !matches!(syn_data_ex.udp_ver, UdpVersion::V2 | UdpVersion::V3) {
            return Err(RdpeudpError::invalid_packet(
                "accept",
                "remote does not support RDPEUDP2",
            ));
        }

        let mut conn = Self::new(Side::Server, config);
        conn.state = State::SynReceived;

        let remote_isn = syn_data.initial_sequence_number;
        let local_isn = conn.config.initial_sequence_number;

        // Negotiate MTU: minimum of all advertised values
        let mtu = conn
            .config
            .upstream_mtu
            .min(conn.config.downstream_mtu)
            .min(syn_data.upstream_mtu)
            .min(syn_data.downstream_mtu);

        conn.params = Some(NegotiatedParams {
            local_isn,
            remote_isn,
            mtu,
            log_window_size: conn.config.log_window_size,
        });

        conn.enqueue_syn_ack(remote_isn, now);
        conn.timers.set(Timer::Idle, now + conn.config.idle_timeout);

        Ok(conn)
    }

    fn new(side: Side, config: ConnectionConfig) -> Self {
        Self {
            side,
            state: State::SynSent,
            config,
            params: None,
            send_window: None,
            recv_window: None,
            congestion: CongestionControl::new(),
            loss_detector: LossDetector::new(),
            reliability: ReliabilityController::new(),
            rtt: RttEstimator::new(),
            timers: TimerTable::new(),
            pending_transmits: VecDeque::new(),
            pending_events: VecDeque::new(),
            send_buffer: VecDeque::new(),
            ack_pending: false,
            cwr_pending: false,
            cn_pending: false,
            pending_ack_of_acks: None,
            handshake_datagram: None,
            handshake_retransmits: 0,
            local_timestamp_ref: 0,
            remote_timestamp_ref: 0,
            wire_buf: Vec::with_capacity(1400),
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Public API
    // ════════════════════════════════════════════════════════════════

    /// Enqueue application data for transmission.
    ///
    /// The data will be sent in the next `poll_transmit()` call,
    /// subject to congestion window availability.
    ///
    /// Returns an error if the connection is not established or
    /// the send buffer is full.
    pub fn send(&mut self, data: Vec<u8>) -> Result<(), RdpeudpError> {
        match self.state {
            State::Established => {}
            State::Closed => return Err(RdpeudpError::connection_closed("send")),
            _ => return Err(RdpeudpError::invalid_state("send")),
        }

        self.send_buffer.push_back(data);
        Ok(())
    }

    /// Process a received wire-format datagram.
    ///
    /// For v1 (handshake) datagrams, pass the raw UDP payload bytes.
    /// For v2 (data) packets, pass the raw UDP payload including the
    /// PacketPrefixByte framing.
    ///
    /// The `wire` slice must be mutable because `decode_with_prefix`
    /// performs an in-place byte swap.
    pub fn handle_datagram(&mut self, wire: &mut [u8], now: MonotonicInstant) -> Result<(), RdpeudpError> {
        if self.state == State::Closed {
            return Err(RdpeudpError::connection_closed("handle datagram"));
        }

        // Reset idle timer on any received packet
        self.timers.set(Timer::Idle, now + self.config.idle_timeout);

        match self.state {
            State::SynSent => self.handle_syn_ack(wire, now),
            State::SynReceived => self.handle_final_ack(wire, now),
            State::Established => {
                // A server that missed our final ACK repeats its SYN+ACK, in
                // the v1 format, long after we have moved on to v2.
                if self.is_repeated_syn_ack(wire) {
                    self.resend_handshake_datagram();
                    return Ok(());
                }

                self.handle_v2_packet(wire, now)
            }
            State::Closed => Err(RdpeudpError::connection_closed("handle datagram")),
        }
    }

    /// Whether `wire` is the server repeating the SYN+ACK we already
    /// answered.
    ///
    /// v1 and v2 datagrams share no framing and no discriminator, so this
    /// leans on the initial sequence number: a v2 data packet would have to
    /// carry the exact 32-bit value the server opened with, at the offset
    /// SYNDATA occupies, to be mistaken for one.
    fn is_repeated_syn_ack(&self, wire: &[u8]) -> bool {
        if self.side != Side::Client {
            return false;
        }

        let Some(params) = self.params.as_ref() else {
            return false;
        };

        let Ok(datagram) = decode::<V1Datagram>(wire) else {
            return false;
        };

        datagram.header.flags.contains(V1Flags::SYN | V1Flags::ACK)
            && datagram
                .syn_data
                .is_some_and(|syn_data| syn_data.initial_sequence_number == params.remote_isn)
    }

    /// Retrieve the next outgoing packet to send on the wire.
    ///
    /// Returns `None` when there are no more packets to send.
    /// The caller should call this in a loop until it returns `None`.
    ///
    /// This method also generates new data packets from the send
    /// buffer when congestion window budget is available, processes
    /// retransmissions, and sends standalone ACKs.
    pub fn poll_transmit(&mut self, now: MonotonicInstant) -> Option<Transmit> {
        // A closed connection has nothing left to say. Returning early also
        // keeps the handshake branch below from arming the keep-alive timer
        // after `close` cleared it: `handle_timeout` returns early once closed,
        // so a timer armed here would stay due forever and a caller polling
        // the deadline without checking `is_closed` would spin.
        if self.state == State::Closed {
            return None;
        }

        // First, drain any pre-built transmits (handshake packets)
        if let Some(t) = self.pending_transmits.pop_front() {
            self.timers.set(Timer::KeepAlive, now + self.config.keep_alive_interval);
            return Some(t);
        }

        if self.state != State::Established {
            return None;
        }

        // Priority order:
        // 1. Retransmissions (reliability queue)
        // 2. New data from send buffer
        // 3. Standalone ACK (if needed)
        // 4. Keep-alive probe

        // Try retransmissions
        if let Some(transmit) = self.try_retransmit(now) {
            self.timers.set(Timer::KeepAlive, now + self.config.keep_alive_interval);
            return Some(transmit);
        }

        // Try new data
        if let Some(transmit) = self.try_send_new_data(now) {
            self.timers.set(Timer::KeepAlive, now + self.config.keep_alive_interval);
            return Some(transmit);
        }

        // Standalone ACK
        if self.ack_pending {
            if let Some(transmit) = self.build_standalone_ack(now) {
                self.ack_pending = false;
                self.timers.clear(Timer::AckDelay);
                self.timers.set(Timer::KeepAlive, now + self.config.keep_alive_interval);
                return Some(transmit);
            }
        }

        None
    }

    /// The next time the caller should invoke `handle_timeout()`.
    ///
    /// Returns `None` if no timers are active.
    pub fn poll_timeout(&self) -> Option<MonotonicInstant> {
        self.timers.next_deadline()
    }

    /// Process expired timers at the current time.
    ///
    /// The caller should invoke this when `now >= poll_timeout()`.
    pub fn handle_timeout(&mut self, now: MonotonicInstant) {
        if self.state == State::Closed {
            return;
        }

        let expired: Vec<Timer> = self.timers.expired(now).collect();

        for timer in expired {
            // The idle timer closes the connection, and `close` clears every
            // timer. Handlers later in this list would otherwise still run and
            // re-arm their own timer on a connection that is already gone,
            // leaving a deadline `handle_timeout` will never service because
            // it returns early once closed. A caller that polls the deadline
            // without checking `is_closed` first would then spin.
            if self.state == State::Closed {
                break;
            }

            match timer {
                Timer::Retransmit => self.handle_retransmit_timeout(now),
                Timer::AckDelay => self.handle_ack_delay_timeout(now),
                Timer::Idle => self.handle_idle_timeout(),
                Timer::KeepAlive => self.handle_keep_alive_timeout(now),
            }
        }
    }

    /// Retrieve the next event for the application layer.
    ///
    /// Returns `None` when there are no more events.
    pub fn poll_event(&mut self) -> Option<Event> {
        self.pending_events.pop_front()
    }

    /// Initiate a graceful close of the connection.
    pub fn close(&mut self) {
        if self.state != State::Closed {
            self.state = State::Closed;
            self.timers.clear(Timer::Retransmit);
            self.timers.clear(Timer::AckDelay);
            self.timers.clear(Timer::Idle);
            self.timers.clear(Timer::KeepAlive);
            self.pending_events.push_back(Event::ConnectionClosed);
        }
    }

    /// Whether the connection is fully established and ready for data.
    pub fn is_established(&self) -> bool {
        self.state == State::Established
    }

    /// Whether the connection has been closed.
    pub fn is_closed(&self) -> bool {
        self.state == State::Closed
    }

    /// Current smoothed RTT estimate, if available.
    pub fn srtt(&self) -> Option<Duration> {
        self.rtt.srtt()
    }

    /// Current retransmission timeout.
    pub fn rto(&self) -> Duration {
        self.rtt.rto()
    }

    /// Negotiated MTU (maximum payload per packet), if handshake is complete.
    pub fn mtu(&self) -> Option<u16> {
        self.params.as_ref().map(|p| p.mtu)
    }

    // ════════════════════════════════════════════════════════════════
    // Handshake: SYN generation
    // ════════════════════════════════════════════════════════════════

    /// Build and enqueue the client SYN datagram.
    fn enqueue_syn(&mut self, now: MonotonicInstant) {
        let datagram = V1Datagram {
            header: FecHeader {
                sn_source_ack: 0xFFFF_FFFF, // No prior packet to ACK
                receive_window_size: 1u16 << u16::from(self.config.log_window_size),
                flags: V1Flags::SYN | V1Flags::SYNEX,
            },
            ack_vector: None,
            ack_of_acks: None,
            syn_data: Some(SynDataPayload {
                initial_sequence_number: self.config.initial_sequence_number,
                upstream_mtu: self.config.upstream_mtu,
                downstream_mtu: self.config.downstream_mtu,
            }),
            correlation_id: None,
            syn_data_ex: Some(SynDataExPayload {
                syn_ex_flags: SynExFlags::VERSION_INFO_VALID,
                udp_ver: UdpVersion::V2,
                cookie_hash: None,
            }),
        };

        self.enqueue_handshake(&datagram);

        self.timers.set(Timer::Retransmit, now + self.rtt.rto());
        self.timers.set(Timer::Idle, now + self.config.idle_timeout);
    }

    /// Build and enqueue the server SYN+ACK datagram.
    fn enqueue_syn_ack(&mut self, remote_isn: u32, now: MonotonicInstant) {
        let datagram = V1Datagram {
            header: FecHeader {
                sn_source_ack: remote_isn,
                receive_window_size: 1u16 << u16::from(self.config.log_window_size),
                flags: V1Flags::SYN | V1Flags::ACK | V1Flags::SYNEX,
            },
            // A SYN+ACK acknowledges the client's SYN through snSourceAck
            // alone (3.1.5.1.3); no ACK vector goes on the wire.
            ack_vector: None,
            ack_of_acks: None,
            syn_data: Some(SynDataPayload {
                initial_sequence_number: self.config.initial_sequence_number,
                upstream_mtu: self.config.upstream_mtu,
                downstream_mtu: self.config.downstream_mtu,
            }),
            correlation_id: None,
            syn_data_ex: Some(SynDataExPayload {
                syn_ex_flags: SynExFlags::VERSION_INFO_VALID,
                udp_ver: UdpVersion::V2,
                cookie_hash: None,
            }),
        };

        self.enqueue_handshake(&datagram);

        self.timers.set(Timer::Retransmit, now + self.rtt.rto());
    }

    /// Queue a handshake datagram and keep a copy for retransmission.
    ///
    /// These datagrams have a fixed shape, so the encode cannot fail in
    /// practice. If it somehow does there is no handshake to be had, and
    /// silently dropping the packet would leave the caller waiting on a
    /// connection that will never answer.
    fn enqueue_handshake(&mut self, datagram: &V1Datagram) {
        let Ok(bytes) = encode_vec(datagram) else {
            self.close();
            return;
        };

        self.pending_transmits.push_back(Transmit {
            contents: bytes.clone(),
        });
        self.handshake_datagram = Some(bytes);
        self.handshake_retransmits = 0;
    }

    /// Put the last handshake datagram back on the wire because the peer
    /// repeated the one before it, so ours went missing.
    ///
    /// Not counted against `HANDSHAKE_RETRANSMIT_LIMIT`: that limit counts
    /// datagrams sent "without a response" (3.1.5.4.1), and this one is a
    /// response.
    fn resend_handshake_datagram(&mut self) {
        if let Some(datagram) = self.handshake_datagram.as_ref() {
            self.pending_transmits.push_back(Transmit {
                contents: datagram.clone(),
            });
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Handshake: receiving
    // ════════════════════════════════════════════════════════════════

    /// Client receives SYN+ACK from server.
    fn handle_syn_ack(&mut self, wire: &[u8], now: MonotonicInstant) -> Result<(), RdpeudpError> {
        let datagram: V1Datagram = decode(wire).map_err(RdpeudpError::decode)?;

        // Must have SYN + ACK flags
        if !datagram.header.flags.contains(V1Flags::SYN | V1Flags::ACK) {
            return Err(RdpeudpError::invalid_packet(
                "handle SYN+ACK",
                "expected SYN+ACK during handshake",
            ));
        }

        let syn_data = datagram
            .syn_data
            .as_ref()
            .ok_or_else(|| RdpeudpError::invalid_packet("handle SYN+ACK", "SYN+ACK missing SynData"))?;

        let syn_data_ex = datagram
            .syn_data_ex
            .as_ref()
            .ok_or_else(|| RdpeudpError::invalid_packet("handle SYN+ACK", "SYN+ACK missing SynDataEx"))?;

        if !matches!(syn_data_ex.udp_ver, UdpVersion::V2 | UdpVersion::V3) {
            return Err(RdpeudpError::invalid_packet(
                "handle SYN+ACK",
                "server does not support RDPEUDP2",
            ));
        }

        let local_isn = self.config.initial_sequence_number;
        let remote_isn = syn_data.initial_sequence_number;

        let mtu = self
            .config
            .upstream_mtu
            .min(self.config.downstream_mtu)
            .min(syn_data.upstream_mtu)
            .min(syn_data.downstream_mtu);

        self.params = Some(NegotiatedParams {
            local_isn,
            remote_isn,
            mtu,
            log_window_size: self.config.log_window_size,
        });

        // Send final ACK to complete handshake
        self.enqueue_final_ack(remote_isn);

        // Transition to established
        self.transition_to_established(now);

        Ok(())
    }

    /// Server receives the client's final ACK.
    fn handle_final_ack(&mut self, wire: &[u8], now: MonotonicInstant) -> Result<(), RdpeudpError> {
        let datagram: V1Datagram = decode(wire).map_err(RdpeudpError::decode)?;

        // The client repeats its SYN when our SYN+ACK goes missing. Answer it
        // again rather than reading it as a protocol violation.
        if datagram.header.flags.contains(V1Flags::SYN) {
            self.resend_handshake_datagram();
            return Ok(());
        }

        if !datagram.header.flags.contains(V1Flags::ACK) {
            return Err(RdpeudpError::invalid_packet(
                "handle final ACK",
                "expected ACK during handshake",
            ));
        }

        self.transition_to_established(now);
        Ok(())
    }

    /// Build and enqueue the client's final ACK.
    fn enqueue_final_ack(&mut self, remote_isn: u32) {
        let datagram = V1Datagram {
            header: FecHeader {
                sn_source_ack: remote_isn,
                receive_window_size: 1u16 << u16::from(self.config.log_window_size),
                flags: V1Flags::ACK,
            },
            ack_vector: Some(V1AckVectorHeader {
                elements: vec![V1AckVectorElement {
                    state: VectorElementState::DatagramReceived,
                    length: 1,
                }],
            }),
            ack_of_acks: None,
            syn_data: None,
            correlation_id: None,
            syn_data_ex: None,
        };

        self.enqueue_handshake(&datagram);
    }

    // ════════════════════════════════════════════════════════════════
    // State transitions
    // ════════════════════════════════════════════════════════════════

    /// Transition from handshake to established state.
    ///
    /// Initializes all data transfer components (windows, timers).
    fn transition_to_established(&mut self, now: MonotonicInstant) {
        self.state = State::Established;
        self.timers.clear(Timer::Retransmit);

        // The client keeps its final ACK: nothing acknowledges that ACK, so
        // the only sign it was lost is the server repeating its SYN+ACK. The
        // server, having just received that ACK, has nothing left to repeat.
        if self.side == Side::Server {
            self.handshake_datagram = None;
        }

        let params = self
            .params
            .as_ref()
            .expect("params must be set before transitioning to established");

        // Data sequence numbers start at ISN + 1
        let local_initial_data_seq = u64::from(params.local_isn) + 1;
        let remote_initial_data_seq = u64::from(params.remote_isn) + 1;

        // Channel sequence numbers start at 1
        let initial_channel_seq = 1u64;

        self.send_window = Some(SendWindow::new(
            local_initial_data_seq,
            initial_channel_seq,
            params.log_window_size,
        ));

        self.recv_window = Some(RecvWindow::new(
            remote_initial_data_seq,
            initial_channel_seq,
            params.log_window_size,
        ));

        self.timers.set(Timer::Idle, now + self.config.idle_timeout);
        self.timers.set(Timer::KeepAlive, now + self.config.keep_alive_interval);

        self.pending_events.push_back(Event::Connected);
    }

    // ════════════════════════════════════════════════════════════════
    // V2 packet handling (established state)
    // ════════════════════════════════════════════════════════════════

    /// Process a v2 wire-format packet (with prefix byte framing).
    fn handle_v2_packet(&mut self, wire: &mut [u8], now: MonotonicInstant) -> Result<(), RdpeudpError> {
        let (_prefix, packet_bytes) = decode_with_prefix(wire).map_err(RdpeudpError::prefix)?;

        let packet: V2Packet = decode(packet_bytes).map_err(RdpeudpError::decode)?;

        // Process ACK payload (cumulative acknowledgment)
        if let Some(ref ack) = packet.ack {
            self.process_ack(ack, &packet.header, now);
        }

        // Process AckVector (selective acknowledgment with gaps)
        if let Some(ref ack_vector) = packet.ack_vector {
            self.process_ack_vector(ack_vector, &packet.header, now);
        }

        // Process AckOfAcks (advance recv window base)
        if let Some(ref aoa) = packet.ack_of_acks {
            self.process_ack_of_acks(aoa);
        }

        // Process CN flag (congestion notification from remote)
        if packet.header.flags.contains(V2Flags::CN) {
            self.process_cn_flag(&packet.header);
        }

        // Process CWR flag (congestion window reduced acknowledgment)
        if packet.header.flags.contains(V2Flags::CWR) {
            // Remote acknowledged our CN: stop setting CN
            self.cn_pending = false;
        }

        // Process data payload
        if let (Some(dh), Some(db)) = (&packet.data_header, &packet.data_body) {
            self.process_data(dh, db, now);
        }

        Ok(())
    }

    /// Process a cumulative ACK payload.
    fn process_ack(&mut self, ack: &AckPayload, _header: &V2Header, now: MonotonicInstant) {
        let send_window = match self.send_window.as_mut() {
            Some(sw) => sw,
            None => return,
        };

        // Update remote timestamp reference from the received timestamp
        self.remote_timestamp_ref = seq::reconstruct_timestamp(ack.received_ts, self.remote_timestamp_ref);

        // Reconstruct the full 64-bit DataSeqNum from the 16-bit wire value
        let reference = send_window.next_data_seq().saturating_sub(1);
        let acked_seq = seq::reconstruct_seq(ack.seq_num, reference);

        // Time the acknowledged packet before acknowledging it. Resolving an
        // entry drains it off the front of the window, so looking it up
        // afterwards finds nothing and no RTT sample is ever taken: the
        // estimator stays empty and the RTO sits at its initial value for
        // the life of the connection.
        //
        // Karn's algorithm: only packets that went out once can be timed.
        let elapsed = send_window
            .get_by_data_seq(acked_seq)
            .filter(|entry| entry.transmit_count == 1)
            .map(|entry| now.duration_since(entry.sent_at));

        // The ACK seq_num is the highest sequentially received DataSeqNum, so
        // everything at or below it is acknowledged.
        let newly_acked_bytes = send_window.mark_received_through(acked_seq);

        if let Some(elapsed) = elapsed {
            let ack_gap = Duration::from_millis(u64::from(ack.send_ack_time_gap));
            if let Some(rtt_sample) = elapsed.checked_sub(ack_gap) {
                self.rtt.update(rtt_sample);
            }
        }

        // Congestion window growth
        if newly_acked_bytes > 0 {
            self.congestion.on_ack(newly_acked_bytes);
        }

        // Run loss detection after processing the ACK
        self.run_loss_detection(now);

        // Update retransmit timer
        self.update_retransmit_timer(now);

        // Generate AckOfAcks to let remote advance their recv window base
        self.pending_ack_of_acks = Some(ack.seq_num);
    }

    /// Process a selective ACK vector.
    fn process_ack_vector(&mut self, ack_vector: &AckVectorPayload, _header: &V2Header, now: MonotonicInstant) {
        let send_window = match self.send_window.as_mut() {
            Some(sw) => sw,
            None => return,
        };

        let reference = send_window.next_data_seq().saturating_sub(1);
        let base = seq::reconstruct_seq(ack_vector.base_seq_num, reference);

        // Walk the ack vector entries and mark packets received/not-received
        let mut current_seq = base;
        let mut newly_acked_bytes: u64 = 0;

        // Timing for the highest packet this vector acknowledges, taken as we
        // go because resolving an entry drains it out of the window. Karn's
        // algorithm: only packets that went out once can be timed.
        let mut elapsed = None;

        for entry in &ack_vector.entries {
            match entry {
                AckVectorEntry::RunLength { received, length } => {
                    for _ in 0..u64::from(*length) {
                        if *received {
                            if let Some(sample) = send_window
                                .get_by_data_seq(current_seq)
                                .filter(|entry| entry.transmit_count == 1)
                                .map(|entry| now.duration_since(entry.sent_at))
                            {
                                elapsed = Some(sample);
                            }

                            if let Some(size) = send_window.mark_received(current_seq) {
                                newly_acked_bytes += u64::try_from(size).expect("packet size fits in u64");
                            }
                        }
                        current_seq += 1;
                    }
                }
                AckVectorEntry::StateMap { bitmap } => {
                    // [MS-RDPEUDP2] 2.2.1.2.6 state-map mode: the most
                    // significant bit is the mode marker and the remaining
                    // seven carry one sequence number each, so a state-map byte
                    // covers seven, not eight. Advancing by eight desynchronised
                    // the cursor from the peer's vector for every entry after
                    // the first state map.
                    for bit_pos in 0..7u32 {
                        let received = (bitmap >> bit_pos) & 1 == 1;
                        if received {
                            if let Some(sample) = send_window
                                .get_by_data_seq(current_seq)
                                .filter(|entry| entry.transmit_count == 1)
                                .map(|entry| now.duration_since(entry.sent_at))
                            {
                                elapsed = Some(sample);
                            }

                            if let Some(size) = send_window.mark_received(current_seq) {
                                newly_acked_bytes += u64::try_from(size).expect("packet size fits in u64");
                            }
                        }
                        current_seq += 1;
                    }
                }
            }
        }

        // RTT estimation from ACKVEC timing
        if let (Some(elapsed), Some(gap_ms)) = (elapsed, ack_vector.send_ack_time_gap_ms) {
            let ack_gap = Duration::from_millis(u64::from(gap_ms));
            if let Some(rtt_sample) = elapsed.checked_sub(ack_gap) {
                self.rtt.update(rtt_sample);
            }
        }

        if newly_acked_bytes > 0 {
            self.congestion.on_ack(newly_acked_bytes);
        }

        self.run_loss_detection(now);
        self.update_retransmit_timer(now);
        self.pending_ack_of_acks = Some(ack_vector.base_seq_num);
    }

    /// Process an AckOfAcks payload: advance our recv window base.
    fn process_ack_of_acks(&mut self, aoa: &AckOfAcksPayload) {
        let recv_window = match self.recv_window.as_mut() {
            Some(rw) => rw,
            None => return,
        };

        let reference = recv_window.highest_seq();
        let new_base = seq::reconstruct_seq(aoa.ack_of_acks_seq_num, reference);
        recv_window.advance_base(new_base);
    }

    /// Process the CN flag from a received packet.
    fn process_cn_flag(&mut self, _header: &V2Header) {
        // The remote is telling us it has detected loss in our direction.
        // React by halving the congestion window.
        let send_window = match self.send_window.as_ref() {
            Some(sw) => sw,
            None => return,
        };

        let ack_seq = send_window.highest_acked_data_seq().unwrap_or(0);
        if self.congestion.on_congestion_notification(ack_seq) {
            // We reduced our window: need to send CWR on next data packet
            self.cwr_pending = true;
        }
    }

    /// Process received data.
    fn process_data(&mut self, dh: &DataHeader, db: &DataBody, now: MonotonicInstant) {
        let recv_window = match self.recv_window.as_mut() {
            Some(rw) => rw,
            None => return,
        };

        // Reconstruct full sequence numbers from 16-bit wire values
        let reference_data = recv_window.highest_seq();
        let data_seq = seq::reconstruct_seq(dh.data_seq_num, reference_data);

        let reference_channel = recv_window.next_channel_seq();
        let channel_seq = seq::reconstruct_seq(db.channel_seq_num, reference_channel);

        // Record the packet in the receive window
        let is_new = recv_window.receive(data_seq, channel_seq, db.data.clone());

        if is_new {
            // Check for gaps: if the receiver has gaps, set CN on our next ACK
            if recv_window.has_gaps() {
                self.cn_pending = true;
            }

            // Drain ordered data for application delivery
            let delivered = recv_window.drain_ordered();
            for chunk in delivered {
                self.pending_events.push_back(Event::DataReceived(chunk));
            }

            // Schedule ACK (either piggybacked on next data or standalone)
            self.ack_pending = true;
            if !self.timers.is_set(Timer::AckDelay) {
                self.timers.set(Timer::AckDelay, now + self.config.ack_delay_timeout);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Loss detection and retransmission
    // ════════════════════════════════════════════════════════════════

    /// Run the loss detector on the current send window state.
    fn run_loss_detection(&mut self, now: MonotonicInstant) {
        let send_window = match self.send_window.as_mut() {
            Some(sw) => sw,
            None => return,
        };

        let highest_acked = send_window.highest_acked_data_seq();
        let rto = self.rtt.rto();

        let lost_seqs = self.loss_detector.detect(send_window, highest_acked, rto, now);

        for data_seq in lost_seqs {
            self.declare_lost(data_seq);
        }
    }

    /// Move one packet out of the send window and onto the retransmit queue.
    fn declare_lost(&mut self, data_seq: u64) {
        let Some(send_window) = self.send_window.as_mut() else {
            return;
        };

        // Read before `mark_lost` drops the entry: this bounds the recovery
        // epoch, so everything already in flight counts as one loss event.
        let largest_sent = send_window.next_data_seq().saturating_sub(1);

        let Some(info) = send_window.mark_lost(data_seq) else {
            return;
        };

        self.congestion.on_loss(data_seq, largest_sent);
        self.reliability.enqueue(info.channel_seq, info.data);

        // Retransmissions always carry CWR.
        self.cwr_pending = true;
    }

    /// Declare the oldest unacknowledged packet lost because the retransmit
    /// timer expired on it.
    ///
    /// The timer firing is itself the loss signal: [MS-RDPEUDP] 3.1.6.1 arms
    /// it for a datagram that has gone unacknowledged for an RTO, and 3.1.1.5
    /// leaves the choice of retransmit scheme to the implementation. Without
    /// this, a packet lost with fewer than `DEFAULT_REORDER_THRESHOLD` later
    /// packets behind it is never retransmitted at all, because the time
    /// threshold in `LossDetector` is computed from an RTO that
    /// `RttEstimator::on_timeout` has already doubled and so always sits
    /// ahead of the elapsed time.
    fn declare_oldest_pending_lost(&mut self) {
        let Some(send_window) = self.send_window.as_ref() else {
            return;
        };

        let Some(oldest) = send_window.pending_entries().map(|entry| entry.data_seq).min() else {
            return;
        };

        self.declare_lost(oldest);
    }

    /// Update the retransmit timer based on send window state.
    fn update_retransmit_timer(&mut self, now: MonotonicInstant) {
        let has_pending = match self.send_window.as_ref() {
            Some(send_window) => send_window.pending_entries().next().is_some(),
            None => return,
        };

        // The retransmit queue counts too. A packet declared lost leaves the
        // send window immediately but may sit in the queue while the
        // congestion window is full, and with no timer armed nothing would
        // come back to it if the peer had also gone quiet.
        if has_pending || self.reliability.has_pending() {
            // Keep the timer running
            if !self.timers.is_set(Timer::Retransmit) {
                self.timers.set(Timer::Retransmit, now + self.rtt.rto());
            }
        } else {
            // Nothing outstanding: clear the timer
            self.timers.clear(Timer::Retransmit);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Transmit generation
    // ════════════════════════════════════════════════════════════════

    /// Try to build and send a retransmission from the reliability queue.
    fn try_retransmit(&mut self, now: MonotonicInstant) -> Option<Transmit> {
        if !self.reliability.has_pending() {
            return None;
        }

        let send_window = self.send_window.as_mut()?;

        // Check congestion window allows sending
        if send_window.bytes_in_flight() >= self.congestion.window() {
            return None;
        }

        if !send_window.has_capacity() {
            return None;
        }

        let entry = self.reliability.dequeue()?;

        // Create a new DataSeqNum for the retransmit but preserve ChannelSeqNum
        let new_data_seq = send_window.push_retransmit(entry.channel_seq, entry.data.clone(), now)?;

        // Build the v2 packet with the retransmitted data
        // Retransmissions always carry CWR
        let transmit = self.build_data_packet(
            new_data_seq,
            entry.channel_seq,
            entry.data,
            true, // CWR on retransmit
            now,
        );

        if transmit.is_some() {
            // Record CWR seq
            self.congestion.set_cwr_seq(new_data_seq);
            self.cwr_pending = false;

            // Reset retransmit timer
            self.timers.set(Timer::Retransmit, now + self.rtt.rto());
        }

        transmit
    }

    /// Try to build and send new data from the send buffer.
    fn try_send_new_data(&mut self, now: MonotonicInstant) -> Option<Transmit> {
        if self.send_buffer.is_empty() {
            return None;
        }

        let send_window = self.send_window.as_mut()?;

        // Check congestion window
        if send_window.bytes_in_flight() >= self.congestion.window() {
            return None;
        }

        if !send_window.has_capacity() {
            return None;
        }

        let data = self.send_buffer.pop_front()?;

        // Push into send window (assigns DataSeqNum and ChannelSeqNum)
        let (data_seq, channel_seq) = send_window.push(data.clone(), now)?;

        let set_cwr = self.cwr_pending;
        let transmit = self.build_data_packet(data_seq, channel_seq, data, set_cwr, now);

        if transmit.is_some() {
            if set_cwr {
                self.congestion.set_cwr_seq(data_seq);
                self.cwr_pending = false;
            }

            // Set retransmit timer if not already running
            if !self.timers.is_set(Timer::Retransmit) {
                self.timers.set(Timer::Retransmit, now + self.rtt.rto());
            }
        }

        transmit
    }

    /// Build a V2 data packet with optional ACK piggybacking.
    fn build_data_packet(
        &mut self,
        data_seq: u64,
        channel_seq: u64,
        data: Vec<u8>,
        set_cwr: bool,
        now: MonotonicInstant,
    ) -> Option<Transmit> {
        let log_window_size = self.params.as_ref()?.log_window_size;

        let mut flags = V2Flags::DATA;

        if set_cwr {
            flags |= V2Flags::CWR;
        }
        if self.cn_pending {
            flags |= V2Flags::CN;
        }

        // Piggyback an acknowledgment if one is pending
        let (ack, ack_vector) = if self.ack_pending {
            self.ack_pending = false;
            self.timers.clear(Timer::AckDelay);

            match self.take_acknowledgement(now) {
                Some(acknowledgement) => {
                    flags |= acknowledgement.flag;
                    (acknowledgement.ack, acknowledgement.ack_vector)
                }
                None => (None, None),
            }
        } else {
            (None, None)
        };

        // AckOfAcks if pending
        let ack_of_acks = self.pending_ack_of_acks.take().map(|seq| AckOfAcksPayload {
            ack_of_acks_seq_num: seq,
        });
        if ack_of_acks.is_some() {
            flags |= V2Flags::AOA;
        }

        let packet = V2Packet {
            header: V2Header { flags, log_window_size },
            ack,
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks,
            data_header: Some(DataHeader {
                data_seq_num: seq::truncate_seq(data_seq),
            }),
            ack_vector,
            data_body: Some(DataBody {
                channel_seq_num: seq::truncate_seq(channel_seq),
                data,
            }),
        };

        self.encode_v2_packet(&packet)
    }

    /// Build a standalone ACK (no data).
    fn build_standalone_ack(&mut self, now: MonotonicInstant) -> Option<Transmit> {
        let log_window_size = self.params.as_ref()?.log_window_size;

        let acknowledgement = self.take_acknowledgement(now)?;
        let mut flags = acknowledgement.flag;
        let (ack, ack_vector) = (acknowledgement.ack, acknowledgement.ack_vector);

        if self.cn_pending {
            flags |= V2Flags::CN;
        }

        let ack_of_acks = self.pending_ack_of_acks.take().map(|seq| AckOfAcksPayload {
            ack_of_acks_seq_num: seq,
        });
        if ack_of_acks.is_some() {
            flags |= V2Flags::AOA;
        }

        let packet = V2Packet {
            header: V2Header { flags, log_window_size },
            ack,
            overhead_size: None,
            delay_ack_info: None,
            ack_of_acks,
            data_header: None,
            ack_vector,
            data_body: None,
        };

        self.encode_v2_packet(&packet)
    }

    /// Build the cumulative ACK payload from the recv window state.
    /// Build the acknowledgment to attach to an outgoing packet, and
    /// release the receive-window slots it covers.
    ///
    /// A cumulative ACK claims every packet through its sequence number, so
    /// it is only true when the window has no holes. [MS-RDPEUDP2] 3.1.1.2.2
    /// keeps the vector form for the other case. Sending the cumulative form
    /// over a hole tells the sender to stop retransmitting a packet that
    /// never arrived, and the receiver never asks for it again.
    fn take_acknowledgement(&mut self, now: MonotonicInstant) -> Option<Acknowledgement> {
        let recv_window = self.recv_window.as_ref()?;

        let acknowledgement = if recv_window.has_gaps() {
            Acknowledgement {
                flag: V2Flags::ACKVEC,
                ack: None,
                ack_vector: Some(self.build_ack_vector_payload(recv_window, now)),
            }
        } else {
            Acknowledgement {
                flag: V2Flags::ACK,
                ack: Some(self.build_ack_payload(recv_window, now)),
                ack_vector: None,
            }
        };

        self.recv_window
            .as_mut()
            .expect("checked at the top of this function")
            .release_acknowledged();

        Some(acknowledgement)
    }

    fn build_ack_payload(&self, recv_window: &RecvWindow, _now: MonotonicInstant) -> AckPayload {
        // seq_num = highest sequentially received DataSeqNum
        // For cumulative ACK, this is base_seq + count of contiguous received - 1,
        // but recv_window.highest_seq() gives us the overall highest.
        // For a true cumulative ACK without gaps, highest_seq is correct.
        // When gaps exist, we use ACKVEC instead.
        let cumulative_seq = recv_window.highest_seq();

        AckPayload {
            seq_num: seq::truncate_seq(cumulative_seq),
            received_ts: seq::truncate_timestamp(self.local_timestamp_ref),
            send_ack_time_gap: 0, // Simplified, real implementation would track processing delay
            delay_ack_time_scale: 0,
            delay_ack_time_additions: Vec::new(),
        }
    }

    /// Build the selective ACK vector payload from recv window state.
    fn build_ack_vector_payload(&self, recv_window: &RecvWindow, _now: MonotonicInstant) -> AckVectorPayload {
        let ack_vec = recv_window.ack_vector();
        let mut entries: Vec<AckVectorEntry> = Vec::new();

        // RunLength entries have a 6-bit length field (max 63).
        // Split longer runs into multiple entries.
        for (received, count) in ack_vec {
            let mut remaining = count;
            while remaining > 0 {
                let chunk = u8::try_from(remaining.min(63)).expect("clamped to 6-bit range");
                entries.push(AckVectorEntry::RunLength {
                    received,
                    length: chunk,
                });
                remaining -= u64::from(chunk);
            }
        }

        AckVectorPayload {
            base_seq_num: seq::truncate_seq(recv_window.base_seq()),
            timestamp: Some(seq::truncate_timestamp(self.local_timestamp_ref)),
            send_ack_time_gap_ms: Some(0),
            entries,
        }
    }

    /// Encode a V2 packet with the PacketPrefixByte framing.
    fn encode_v2_packet(&mut self, packet: &V2Packet) -> Option<Transmit> {
        let packet_bytes = encode_vec(packet).ok()?;
        let is_dummy = packet.header.flags.contains(V2Flags::DUMMY);

        self.wire_buf.clear();
        encode_with_prefix(&packet_bytes, is_dummy, &mut self.wire_buf).ok()?;

        Some(Transmit {
            contents: self.wire_buf.clone(),
        })
    }

    // ════════════════════════════════════════════════════════════════
    // Timer handlers
    // ════════════════════════════════════════════════════════════════

    /// Handle retransmit timer expiry.
    fn handle_retransmit_timeout(&mut self, now: MonotonicInstant) {
        self.timers.clear(Timer::Retransmit);

        if self.state != State::Established {
            self.rtt.on_timeout();
            self.retransmit_handshake(now);
            return;
        }

        // Before the backoff, which moves the threshold this would be
        // measured against out of reach. See `declare_oldest_pending_lost`.
        self.declare_oldest_pending_lost();

        // Apply exponential backoff
        self.rtt.on_timeout();

        // Reset the timer if there is still unfinished business
        self.update_retransmit_timer(now);
    }

    /// Resend the handshake datagram the peer has not answered, or close
    /// once it has had enough chances (3.1.5.4.1).
    fn retransmit_handshake(&mut self, now: MonotonicInstant) {
        if self.handshake_datagram.is_none() {
            return;
        }

        if self.handshake_retransmits >= HANDSHAKE_RETRANSMIT_LIMIT {
            self.close();
            return;
        }

        self.handshake_retransmits += 1;
        self.resend_handshake_datagram();

        // 3.1.6.1 wants the timer to keep firing at no less than the same
        // interval; `on_timeout` above has already backed the RTO off.
        self.timers.set(Timer::Retransmit, now + self.rtt.rto());
    }

    /// Handle ACK delay timer expiry: send a standalone ACK.
    fn handle_ack_delay_timeout(&mut self, _now: MonotonicInstant) {
        self.timers.clear(Timer::AckDelay);
        self.ack_pending = true;
    }

    /// Handle idle timeout: close the connection.
    fn handle_idle_timeout(&mut self) {
        self.close();
    }

    /// Handle keep-alive timer: enqueue a keep-alive probe.
    fn handle_keep_alive_timeout(&mut self, now: MonotonicInstant) {
        self.timers.clear(Timer::KeepAlive);

        // Send a standalone ACK as a keep-alive probe
        self.ack_pending = true;

        self.timers.set(Timer::KeepAlive, now + self.config.keep_alive_interval);
    }
}

impl core::fmt::Debug for RdpeudpConnection {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RdpeudpConnection")
            .field("side", &self.side)
            .field("state", &self.state)
            .field("rto", &self.rtt.rto())
            .field("srtt", &self.rtt.srtt())
            .field("pending_transmits", &self.pending_transmits.len())
            .field("pending_events", &self.pending_events.len())
            .field("send_buffer", &self.send_buffer.len())
            .finish_non_exhaustive()
    }
}
