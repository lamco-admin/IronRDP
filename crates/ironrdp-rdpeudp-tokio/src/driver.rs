//! Driver task: the async event loop bridging UDP socket ↔ sans-I/O state machine.
//!
//! A single `tokio::spawn`'d task owns the `UdpSocket` and `RdpeudpConnection`,
//! running a `select!` loop over three event sources:
//!
//! 1. **UDP recv**: incoming datagrams from the network
//! 2. **Write data**: the TLS layer has plaintext to transmit
//! 3. **Timer**: RDPEUDP2 retransmit / ACK-delay / keep-alive / idle timeouts
//!
//! The driver communicates with the `RdpeudpStream` (which tokio-rustls wraps)
//! through `Arc<Mutex<SharedIo>>`.

use crate::clock::Clock;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::io;
use std::sync::{Arc, Mutex};

use ironrdp_rdpeudp::{Event, RdpeudpConnection, RdpeudpError, RdpeudpErrorKind};
use tokio::net::UdpSocket;
use tokio::sync::Notify;

use crate::error::{DriverError, DriverErrorExt as _, DriverErrorKind};
use crate::stream::SharedIo;

/// Maximum UDP datagram size we'll attempt to receive.
/// RDPEUDP2 typically negotiates 1232-byte MTU, but we budget
/// a full Ethernet jumbo frame for safety.
const RECV_BUF_SIZE: usize = 9000;

// ════════════════════════════════════════════════════════════════════
// Driver
// ════════════════════════════════════════════════════════════════════

/// The driver task's internal state.
pub(crate) struct Driver {
    socket: UdpSocket,
    conn: RdpeudpConnection,
    shared: Arc<Mutex<SharedIo>>,
    /// Notified by the driver when Event::Connected fires.
    connected_notify: Arc<Notify>,
    /// Whether Event::Connected has already been signaled.
    connected_signaled: bool,
    /// Receive buffer (reused across iterations).
    recv_buf: Vec<u8>,
    /// The only clock in the stack; the state machine has none.
    clock: Clock,
}

impl Driver {
    pub(crate) fn new(
        socket: UdpSocket,
        conn: RdpeudpConnection,
        shared: Arc<Mutex<SharedIo>>,
        connected_notify: Arc<Notify>,
    ) -> Self {
        Self {
            socket,
            conn,
            shared,
            connected_notify,
            connected_signaled: false,
            recv_buf: vec![0u8; RECV_BUF_SIZE],
            clock: Clock::new(),
        }
    }

    /// Run the driver event loop until the connection closes or errors.
    pub(crate) async fn run(mut self) -> Result<(), DriverError> {
        let result = self.run_to_completion().await;

        // However this ended, the stream side has to hear about it. A reader
        // parked in `poll_read` has left its waker here and nothing else will
        // poll it, so returning an error without coming through here hangs
        // that task for the life of the process.
        self.release_stream(result.as_ref().err());

        result
    }

    /// Mark the shared state closed and wake everything waiting on it.
    fn release_stream(&self, error: Option<&DriverError>) {
        let Ok(mut shared) = self.shared.lock() else {
            // A poisoned lock means the stream side already panicked, and
            // there is nothing left to wake.
            return;
        };

        if let Some(error) = error {
            let kind = match error.kind() {
                DriverErrorKind::Socket(io_error) => io_error.kind(),
                DriverErrorKind::Rdpeudp(_) => io::ErrorKind::InvalidData,
                DriverErrorKind::ConnectionClosed => io::ErrorKind::ConnectionAborted,
            };
            shared.error.get_or_insert(kind);
        }

        shared.closed = true;

        for waker in [
            shared.read_waker.take(),
            shared.write_waker.take(),
            shared.flush_waker.take(),
        ]
        .into_iter()
        .flatten()
        {
            waker.wake();
        }
    }

    async fn run_to_completion(&mut self) -> Result<(), DriverError> {
        // Send any initial transmits (the SYN packet for client-side connections)
        self.drain_transmits().await?;

        loop {
            let timeout = self
                .conn
                .poll_timeout()
                .map(|deadline| tokio::time::Instant::from_std(self.clock.deadline(deadline)));

            tokio::select! {
                biased;

                // Branch 1: Incoming UDP datagram (highest priority)
                result = self.socket.recv(&mut self.recv_buf) => {
                    let n = result.map_err(|error| DriverError::socket("receive datagram", error))?;
                    let now = self.clock.now();

                    // handle_datagram takes &mut [u8] for in-place prefix byte swap
                    match self.conn.handle_datagram(&mut self.recv_buf[..n], now) {
                        Ok(()) => {}
                        // One datagram the state machine cannot make sense of
                        // is not grounds for tearing down the connection. UDP
                        // delivers whatever arrives at the socket, including
                        // corruption and anything an off-path sender puts
                        // there, and a peer whose retransmitted handshake
                        // datagram lands late produces this too. Drop it and
                        // carry on; a genuinely dead connection still closes
                        // on the idle timeout.
                        Err(error) if is_droppable(&error) => {
                            tracing::debug!(%error, "dropping an unusable datagram");
                        }
                        Err(error) => return Err(DriverError::rdpeudp("handle datagram", error)),
                    }

                    self.drain_transmits().await?;
                    self.drain_events();
                }

                // Branch 2: TLS layer has data to send
                _ = WriteDataReady::new(&self.shared) => {
                    let (data, stream_closed, flush_waker) = {
                        let mut shared = self.shared.lock()
                            .map_err(|_| DriverError::connection_closed("lock shared state"))?;
                        let closed = shared.closed && shared.write_buf.is_empty();
                        let buf = shared.write_buf.split().freeze().to_vec();
                        let flush = shared.flush_waker.take();
                        (buf, closed, flush)
                    };
                    // Wake any pending flush (write_buf has been drained)
                    if let Some(waker) = flush_waker {
                        waker.wake();
                    }
                    if stream_closed {
                        self.conn.close();
                        self.drain_events();
                        return Ok(());
                    }
                    if !data.is_empty() {
                        self.conn
                            .send(data)
                            .map_err(|error| DriverError::rdpeudp("queue outbound data", error))?;
                        self.drain_transmits().await?;
                    }
                }

                // Branch 3: Timer expiry
                _ = optional_sleep(timeout) => {
                    let now = self.clock.now();
                    self.conn.handle_timeout(now);
                    self.drain_transmits().await?;
                    self.drain_events();
                }
            }

            if self.conn.is_closed() {
                // Propagate close to the stream side
                let mut shared = self
                    .shared
                    .lock()
                    .map_err(|_| DriverError::connection_closed("lock shared state"))?;
                shared.closed = true;
                if let Some(waker) = shared.read_waker.take() {
                    waker.wake();
                }
                return Ok(());
            }
        }
    }

    /// Send all pending transmits to the UDP socket.
    async fn drain_transmits(&mut self) -> Result<(), DriverError> {
        let now = self.clock.now();
        while let Some(transmit) = self.conn.poll_transmit(now) {
            self.socket
                .send(&transmit.contents)
                .await
                .map_err(|error| DriverError::socket("send datagram", error))?;
        }
        Ok(())
    }

    /// Process all pending events from the connection state machine.
    fn drain_events(&mut self) {
        while let Some(event) = self.conn.poll_event() {
            match event {
                Event::Connected => {
                    if !self.connected_signaled {
                        self.connected_signaled = true;
                        self.connected_notify.notify_one();
                    }
                }
                Event::DataReceived(data) => {
                    if let Ok(mut shared) = self.shared.lock() {
                        shared.read_buf.extend_from_slice(&data);
                        if let Some(waker) = shared.read_waker.take() {
                            waker.wake();
                        }
                    }
                }
                Event::ConnectionClosed => {
                    if let Ok(mut shared) = self.shared.lock() {
                        shared.closed = true;
                        if let Some(waker) = shared.read_waker.take() {
                            waker.wake();
                        }
                    }
                }
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Helper futures
// ════════════════════════════════════════════════════════════════════

/// Future that resolves when `SharedIo::write_buf` has data OR when
/// the stream is shut down.
///
/// Cancel-safe: only registers a waker, no side effects on drop.
struct WriteDataReady {
    shared: Arc<Mutex<SharedIo>>,
}

impl WriteDataReady {
    fn new(shared: &Arc<Mutex<SharedIo>>) -> Self {
        Self {
            shared: Arc::clone(shared),
        }
    }
}

impl Future for WriteDataReady {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let Ok(mut shared) = self.shared.lock() else {
            return Poll::Ready(());
        };

        if !shared.write_buf.is_empty() || shared.closed {
            return Poll::Ready(());
        }

        shared.write_waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

/// Sleep until the given deadline, or pend forever if `None`.
async fn optional_sleep(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => core::future::pending().await,
    }
}

/// Whether a datagram that the state machine rejected can simply be dropped.
///
/// Anything the peer could put on the wire, deliberately or by accident, is
/// droppable: a malformed packet, a packet that makes no sense in the current
/// state. What is not droppable is the state machine reporting that this end
/// is finished, which is a real reason to stop the loop.
fn is_droppable(error: &RdpeudpError) -> bool {
    matches!(
        error.kind(),
        RdpeudpErrorKind::Decode(_)
            | RdpeudpErrorKind::Prefix(_)
            | RdpeudpErrorKind::InvalidPacket { .. }
            | RdpeudpErrorKind::InvalidState
    )
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use ironrdp_rdpeudp::ConnectionConfig;

    /// `connect` requires the cookie hash a version 3 SYN carries. These tests
    /// never reach a real multitransport request, so any value will do.
    fn test_connection_config() -> ConnectionConfig {
        ConnectionConfig {
            cookie_hash: Some([0x5A; 32]),
            ..ConnectionConfig::default()
        }
    }
    use ironrdp_rdpeudp::RdpeudpErrorExt as _;

    use super::*;

    #[tokio::test]
    async fn driver_sends_initial_syn() {
        // Bind two UDP sockets on localhost to simulate client/server
        let client_sock = UdpSocket::bind("127.0.0.1:0").await.expect("bind client");
        let server_sock = UdpSocket::bind("127.0.0.1:0").await.expect("bind server");

        let server_addr = server_sock.local_addr().expect("server addr");
        client_sock.connect(server_addr).await.expect("connect");

        let conn = RdpeudpConnection::connect(test_connection_config(), Clock::new().now()).expect("connect");

        let shared = Arc::new(Mutex::new(SharedIo::new()));
        let notify = Arc::new(Notify::new());
        let driver = Driver::new(client_sock, conn, shared, notify);

        // Run the driver briefly: it should send the SYN packet
        let handle = tokio::spawn(driver.run());

        // Receive the SYN on the server side
        let mut buf = [0u8; 1500];
        let recv_result = tokio::time::timeout(Duration::from_secs(2), server_sock.recv(&mut buf)).await;

        assert!(recv_result.is_ok(), "should receive SYN packet");
        let n = recv_result.expect("timeout").expect("recv");
        assert!(n > 0, "SYN packet should have content");

        // Abort the driver (we only needed to test the initial SYN)
        handle.abort();
    }

    /// Anything the peer can put on the wire is droppable; our own state
    /// machine telling us this end is finished is not.
    #[test]
    fn droppable_errors_are_the_ones_the_peer_controls() {
        assert!(is_droppable(&RdpeudpError::invalid_packet("test", "malformed")));
        assert!(!is_droppable(&RdpeudpError::connection_closed("test")));
    }

    /// A driver that exits on an error has to release the stream side. A
    /// reader parked in `poll_read` left its waker in the shared state and
    /// nothing else is going to poll it.
    #[tokio::test]
    async fn an_error_exit_wakes_a_parked_reader() {
        let socket = UdpSocket::bind("127.0.0.1:0").await.expect("bind");
        let conn = RdpeudpConnection::connect(test_connection_config(), Clock::new().now()).expect("connect");

        let shared = Arc::new(Mutex::new(SharedIo::new()));
        let woken = Arc::new(core::sync::atomic::AtomicBool::new(false));

        {
            let mut guard = shared.lock().expect("lock");
            guard.read_waker = Some(futures_waker(Arc::clone(&woken)));
        }

        let driver = Driver::new(socket, conn, Arc::clone(&shared), Arc::new(Notify::new()));
        driver.release_stream(Some(&DriverError::connection_closed("test")));

        assert!(
            woken.load(core::sync::atomic::Ordering::SeqCst),
            "the reader was left parked"
        );

        let guard = shared.lock().expect("lock");
        assert!(guard.closed);
        assert_eq!(guard.error, Some(io::ErrorKind::ConnectionAborted));
    }

    /// Build a waker that records having been woken.
    fn futures_waker(flag: Arc<core::sync::atomic::AtomicBool>) -> core::task::Waker {
        struct Flag(Arc<core::sync::atomic::AtomicBool>);

        impl std::task::Wake for Flag {
            fn wake(self: Arc<Self>) {
                self.0.store(true, core::sync::atomic::Ordering::SeqCst);
            }
        }

        core::task::Waker::from(Arc::new(Flag(flag)))
    }
}
