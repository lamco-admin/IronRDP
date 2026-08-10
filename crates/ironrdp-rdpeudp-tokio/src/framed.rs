//! `FramedRead` + `FramedWrite` trait implementations for `UdpTransport`.
//!
//! These allow `UdpTransport` to be used with `ironrdp-async`'s `Framed`
//! infrastructure, so the session layer can read/write DVC frames over
//! the UDP tunnel using the same trait interface as the TCP path.
//!
//! Unlike TCP (which is a byte stream requiring `find_size()` framing),
//! UDP tunnel data arrives as complete RDPEMT-deframed messages. Each
//! `read()` appends exactly one message to the buffer.

use core::pin::Pin;
use std::io;

use bytes::BytesMut;
use ironrdp_async::{FramedRead, FramedWrite};

use crate::transport::UdpTransport;

impl FramedRead for UdpTransport {
    type ReadFut<'read>
        = Pin<Box<dyn Future<Output = io::Result<usize>> + Send + Sync + 'read>>
    where
        Self: 'read;

    fn read<'a>(&'a mut self, buf: &'a mut BytesMut) -> Self::ReadFut<'a> {
        Box::pin(async move {
            match self.recv().await {
                Some(data) => {
                    let n = data.len();
                    buf.extend_from_slice(&data);
                    Ok(n)
                }
                None => Ok(0), // EOF: tunnel closed
            }
        })
    }
}

impl FramedWrite for UdpTransport {
    type WriteAllFut<'write>
        = Pin<Box<dyn Future<Output = io::Result<()>> + Send + Sync + 'write>>
    where
        Self: 'write;

    fn write_all<'a>(&'a mut self, buf: &'a [u8]) -> Self::WriteAllFut<'a> {
        Box::pin(async move {
            self.send(buf.to_vec())
                .await
                .map_err(|e| io::Error::new(io::ErrorKind::ConnectionReset, e.to_string()))
        })
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc;

    use super::*;

    /// Build a `UdpTransport` backed by test channels (no real network).
    fn test_transport() -> (UdpTransport, mpsc::Sender<Vec<u8>>, mpsc::Receiver<Vec<u8>>) {
        let (incoming_tx, incoming_rx) = mpsc::channel::<Vec<u8>>(16);
        let (outgoing_tx, outgoing_rx) = mpsc::channel::<Vec<u8>>(16);

        let transport = UdpTransport::from_channels(incoming_rx, outgoing_tx);

        (transport, incoming_tx, outgoing_rx)
    }

    #[tokio::test]
    async fn framed_read_delivers_one_message() {
        let (mut transport, feeder, _) = test_transport();
        feeder.send(vec![0xDE, 0xAD, 0xBE, 0xEF]).await.unwrap();

        let mut buf = BytesMut::new();
        let n = FramedRead::read(&mut transport, &mut buf).await.unwrap();
        assert_eq!(n, 4);
        assert_eq!(&*buf, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[tokio::test]
    async fn framed_read_returns_eof_on_closed_channel() {
        let (mut transport, feeder, _) = test_transport();
        drop(feeder);

        let mut buf = BytesMut::new();
        let n = FramedRead::read(&mut transport, &mut buf).await.unwrap();
        assert_eq!(n, 0);
        assert!(buf.is_empty());
    }

    #[tokio::test]
    async fn framed_write_sends_data() {
        let (mut transport, _, mut receiver) = test_transport();

        FramedWrite::write_all(&mut transport, &[0x01, 0x02, 0x03])
            .await
            .unwrap();

        let data = receiver.recv().await.unwrap();
        assert_eq!(data, vec![0x01, 0x02, 0x03]);
    }

    #[tokio::test]
    async fn framed_write_returns_error_on_closed_channel() {
        let (mut transport, _, receiver) = test_transport();
        drop(receiver);

        let result = FramedWrite::write_all(&mut transport, &[0x01]).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::ConnectionReset);
    }

    #[tokio::test]
    async fn framed_read_multiple_messages_accumulate() {
        let (mut transport, feeder, _) = test_transport();

        feeder.send(vec![0xAA, 0xBB]).await.unwrap();
        feeder.send(vec![0xCC, 0xDD]).await.unwrap();

        let mut buf = BytesMut::new();

        let n1 = FramedRead::read(&mut transport, &mut buf).await.unwrap();
        assert_eq!(n1, 2);
        assert_eq!(&*buf, &[0xAA, 0xBB]);

        let n2 = FramedRead::read(&mut transport, &mut buf).await.unwrap();
        assert_eq!(n2, 2);
        assert_eq!(&*buf, &[0xAA, 0xBB, 0xCC, 0xDD]);
    }
}
