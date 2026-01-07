//! EGFX (Graphics Pipeline Extension) server support
//!
//! This module provides:
//! - Factory trait for creating EGFX handlers
//! - Bridge wrapper for shared access to GraphicsPipelineServer
//! - Types for proactive frame sending via ServerEvent
//!
//! # Architecture
//!
//! The EGFX channel uses a "bridge" pattern to enable proactive frame sending:
//!
//! ```text
//! GfxServerFactory
//!       │
//!       ├─► Creates Arc<Mutex<GraphicsPipelineServer>>
//!       │
//!       ├─► Returns handle to display handler (for frame sending)
//!       │
//!       └─► Wraps in GfxDvcBridge for DrdynvcServer (for client messages)
//! ```
//!
//! This allows the display handler to call `send_avc420_frame()` directly
//! while the DVC infrastructure handles client capability negotiation and
//! frame acknowledgments.

use std::sync::{Arc, Mutex};

use ironrdp_core::impl_as_any;
use ironrdp_dvc::{DvcMessage, DvcProcessor, DvcServerProcessor};
use ironrdp_egfx::server::{GraphicsPipelineHandler, GraphicsPipelineServer};
use ironrdp_pdu::PduResult;
use ironrdp_svc::SvcMessage;

/// Handle to a shared GraphicsPipelineServer
///
/// Use this to call methods like `send_avc420_frame()` from outside
/// the DVC processing path.
///
/// Uses `std::sync::Mutex` (not tokio) because the `DvcProcessor` trait
/// has synchronous methods that cannot use async locks.
pub type GfxServerHandle = Arc<Mutex<GraphicsPipelineServer>>;

/// Factory trait for creating EGFX graphics pipeline handlers
///
/// Implementors provide:
/// 1. A handler for EGFX callbacks (capability negotiation, frame acks)
/// 2. Optionally, a shared handle to the GraphicsPipelineServer for proactive frame sending
///
/// # Basic Usage (Handler Only)
///
/// ```ignore
/// impl GfxServerFactory for MyFactory {
///     fn build_gfx_handler(&self) -> Box<dyn GraphicsPipelineHandler> {
///         Box::new(MyHandler::new())
///     }
/// }
/// ```
///
/// # Advanced Usage (With Shared Server Access)
///
/// For proactive frame sending, implement `build_server_with_handle()`:
///
/// ```ignore
/// impl GfxServerFactory for MyFactory {
///     fn build_gfx_handler(&self) -> Box<dyn GraphicsPipelineHandler> {
///         Box::new(MyHandler::new())
///     }
///
///     fn build_server_with_handle(&self) -> Option<(GfxDvcBridge, GfxServerHandle)> {
///         let handler = Box::new(MyHandler::new());
///         let server = Arc::new(Mutex::new(GraphicsPipelineServer::new(handler)));
///         let bridge = GfxDvcBridge::new(Arc::clone(&server));
///         Some((bridge, server))
///     }
/// }
/// ```
pub trait GfxServerFactory: Send {
    /// Create a new graphics pipeline handler
    ///
    /// This is used when shared server access is not needed.
    fn build_gfx_handler(&self) -> Box<dyn GraphicsPipelineHandler>;

    /// Create a bridge and shared server handle
    ///
    /// Override this method to enable proactive frame sending.
    /// Returns `None` by default, which falls back to `build_gfx_handler()`.
    ///
    /// When returning `Some((bridge, handle))`:
    /// - `bridge` is registered with DrdynvcServer (handles client messages)
    /// - `handle` is stored for frame sending (display handler access)
    fn build_server_with_handle(&self) -> Option<(GfxDvcBridge, GfxServerHandle)> {
        None
    }
}

/// Bridge wrapper for shared GraphicsPipelineServer access
///
/// This wrapper implements `DvcProcessor` by delegating to the inner
/// `GraphicsPipelineServer`. It holds an `Arc<Mutex<>>` to enable
/// shared access from both the DVC layer and the display handler.
///
/// # Thread Safety
///
/// The bridge uses `std::sync::Mutex` for synchronous locking.
/// This is required because `DvcProcessor` trait methods are synchronous
/// and cannot use async locks. The mutex is held briefly during each
/// operation, so contention should be minimal.
pub struct GfxDvcBridge {
    inner: GfxServerHandle,
}

impl GfxDvcBridge {
    /// Create a new bridge wrapping a shared GraphicsPipelineServer
    pub fn new(server: GfxServerHandle) -> Self {
        Self { inner: server }
    }

    /// Get a reference to the underlying server handle
    ///
    /// Use this to access the server for frame sending.
    pub fn server(&self) -> &GfxServerHandle {
        &self.inner
    }
}

impl_as_any!(GfxDvcBridge);

impl DvcProcessor for GfxDvcBridge {
    fn channel_name(&self) -> &str {
        ironrdp_egfx::CHANNEL_NAME
    }

    fn start(&mut self, channel_id: u32) -> PduResult<Vec<DvcMessage>> {
        self.inner
            .lock()
            .expect("GfxServerHandle mutex poisoned")
            .start(channel_id)
    }

    fn process(&mut self, channel_id: u32, payload: &[u8]) -> PduResult<Vec<DvcMessage>> {
        self.inner
            .lock()
            .expect("GfxServerHandle mutex poisoned")
            .process(channel_id, payload)
    }

    fn close(&mut self, channel_id: u32) {
        self.inner
            .lock()
            .expect("GfxServerHandle mutex poisoned")
            .close(channel_id)
    }
}

impl DvcServerProcessor for GfxDvcBridge {}

/// Message type for EGFX server events
///
/// These events are sent through `ServerEvent::Egfx` to route
/// EGFX PDUs to the wire without waiting for client messages.
#[derive(Debug)]
pub enum EgfxServerMessage {
    /// Pre-encoded DVC messages ready to send to client
    ///
    /// These are typically the output of `GraphicsPipelineServer::drain_output()`
    /// after calling `send_avc420_frame()` or similar methods.
    SendMessages {
        /// The DVC channel ID for the EGFX channel
        channel_id: u32,
        /// Encoded SVC messages wrapping the DVC data
        messages: Vec<SvcMessage>,
    },
}

impl std::fmt::Display for EgfxServerMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EgfxServerMessage::SendMessages { channel_id, messages } => {
                write!(f, "SendMessages(channel={}, count={})", channel_id, messages.len())
            }
        }
    }
}
