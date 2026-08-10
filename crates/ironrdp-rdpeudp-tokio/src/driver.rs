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
use std::sync::{Arc, Mutex};

use ironrdp_rdpeudp::{Event, RdpeudpConnection};
use tokio::net::UdpSocket;
use tokio::sync::Notify;

use crate::error::{DriverError, DriverErrorExt as _};
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
                    self.conn
                        .handle_datagram(&mut self.recv_buf[..n], now)
                        .map_err(|error| DriverError::rdpeudp("handle datagram", error))?;
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

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use ironrdp_rdpeudp::ConnectionConfig;

    use super::*;

    #[tokio::test]
    async fn driver_sends_initial_syn() {
        // Bind two UDP sockets on localhost to simulate client/server
        let client_sock = UdpSocket::bind("127.0.0.1:0").await.expect("bind client");
        let server_sock = UdpSocket::bind("127.0.0.1:0").await.expect("bind server");

        let server_addr = server_sock.local_addr().expect("server addr");
        client_sock.connect(server_addr).await.expect("connect");

        let config = ConnectionConfig::default();
        let conn = RdpeudpConnection::connect(config, Clock::new().now());

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
}
