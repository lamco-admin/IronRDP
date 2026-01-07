//! Server-side EGFX implementation
//!
//! This module provides complete server-side support for the Graphics Pipeline Extension
//! (MS-RDPEGFX), enabling H.264 AVC420/AVC444 video streaming to RDP clients.
//!
//! # Protocol Compliance
//!
//! This implementation follows MS-RDPEGFX specification requirements:
//!
//! - **Capability Negotiation**: Supports V8, V8.1, V10, V10.1-V10.7
//! - **Surface Management**: Multi-surface support with proper lifecycle
//! - **Frame Flow Control**: Tracks unacknowledged frames per spec
//! - **Codec Support**: AVC420, AVC444, with extensibility for others
//!
//! # Architecture
//!
//! The server follows this message flow:
//!
//! ```text
//! Client                                  Server
//!    |                                       |
//!    |--- CapabilitiesAdvertise ------------>|
//!    |                                       | (negotiate capabilities)
//!    |<----------- CapabilitiesConfirm ------|
//!    |<----------- ResetGraphics ------------|
//!    |<----------- CreateSurface ------------|
//!    |<----------- MapSurfaceToOutput -------|
//!    |                                       |
//!    |  (For each frame:)                    |
//!    |<----------- StartFrame ---------------|
//!    |<----------- WireToSurface1/2 ---------|  (H.264 data)
//!    |<----------- EndFrame -----------------|
//!    |                                       |
//!    |--- FrameAcknowledge ----------------->|  (flow control)
//!    |--- QoeFrameAcknowledge -------------->|  (optional, V10+)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use ironrdp_egfx::server::{GraphicsPipelineServer, GraphicsPipelineHandler};
//!
//! struct MyHandler;
//!
//! impl GraphicsPipelineHandler for MyHandler {
//!     fn capabilities_advertise(&mut self, caps: &CapabilitiesAdvertisePdu) {
//!         // Client sent capabilities
//!     }
//!
//!     fn on_ready(&mut self, negotiated: &CapabilitySet) {
//!         // Server is ready to send frames
//!     }
//! }
//!
//! let server = GraphicsPipelineServer::new(Box::new(MyHandler));
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use ironrdp_core::{decode, impl_as_any, Encode, EncodeResult, WriteCursor};
use ironrdp_dvc::{DvcEncode, DvcMessage, DvcProcessor, DvcServerProcessor};
use ironrdp_graphics::zgfx::{self, CompressionMode, Compressor};
use ironrdp_pdu::gcc::Monitor;
use ironrdp_pdu::geometry::InclusiveRectangle;
use ironrdp_pdu::{decode_err, PduResult};
use tracing::{debug, trace, warn};

use crate::pdu::{
    encode_avc420_bitmap_stream, Avc420BitmapStream, Avc420Region, Avc444BitmapStream, CacheImportOfferPdu,
    CacheImportReplyPdu, CapabilitiesAdvertisePdu, CapabilitiesConfirmPdu, CapabilitiesV103Flags,
    CapabilitiesV104Flags, CapabilitiesV107Flags, CapabilitiesV10Flags, CapabilitiesV81Flags, CapabilitiesV8Flags,
    CapabilitySet, Codec1Type, CreateSurfacePdu, DeleteSurfacePdu, Encoding, EndFramePdu, FrameAcknowledgePdu, GfxPdu,
    MapSurfaceToOutputPdu, PixelFormat, QoeFrameAcknowledgePdu, ResetGraphicsPdu, StartFramePdu, Timestamp,
    WireToSurface1Pdu,
};
use crate::CHANNEL_NAME;

// ============================================================================
// Constants
// ============================================================================

/// Default maximum frames in flight before applying backpressure
const DEFAULT_MAX_FRAMES_IN_FLIGHT: u32 = 3;

/// Special queue depth value indicating client has disabled acknowledgments
const SUSPEND_FRAME_ACK_QUEUE_DEPTH: u32 = 0xFFFFFFFF;

// ============================================================================
// ZGFX Wrapper
// ============================================================================

/// Wrapper that contains pre-encoded ZGFX-wrapped bytes
///
/// This type holds bytes that have already been:
/// 1. Encoded from GfxPdu
/// 2. Optionally compressed with ZGFX
/// 3. Wrapped in ZGFX segment structure
///
/// It implements DvcEncode to return these pre-encoded bytes directly.
///
/// # Why Pre-encode?
///
/// ZGFX compression requires mutable state (history buffer) but the Encode
/// trait's encode() method takes &self. So we pre-compress in drain_output()
/// where we have mutable access to the compressor.
struct ZgfxWrappedBytes {
    bytes: Vec<u8>,
    pdu_name: &'static str,
}

impl ZgfxWrappedBytes {
    fn new(bytes: Vec<u8>, pdu_name: &'static str) -> Self {
        Self { bytes, pdu_name }
    }
}

impl Encode for ZgfxWrappedBytes {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        dst.write_slice(&self.bytes);
        Ok(())
    }

    fn name(&self) -> &'static str {
        self.pdu_name
    }

    fn size(&self) -> usize {
        self.bytes.len()
    }
}

impl DvcEncode for ZgfxWrappedBytes {}

// ============================================================================
// Surface Management
// ============================================================================

/// Surface state tracked by server
///
/// Per MS-RDPEGFX, the server maintains an "Offscreen Surfaces ADM element"
/// which is a list of surfaces created on the client.
#[derive(Debug, Clone)]
pub struct Surface {
    /// Surface identifier (unique per session)
    pub id: u16,
    /// Surface width in pixels
    pub width: u16,
    /// Surface height in pixels
    pub height: u16,
    /// Pixel format
    pub pixel_format: PixelFormat,
    /// Whether this surface is mapped to an output
    pub is_mapped: bool,
    /// Output X origin (if mapped)
    pub output_origin_x: u32,
    /// Output Y origin (if mapped)
    pub output_origin_y: u32,
}

impl Surface {
    fn new(id: u16, width: u16, height: u16, pixel_format: PixelFormat) -> Self {
        Self {
            id,
            width,
            height,
            pixel_format,
            is_mapped: false,
            output_origin_x: 0,
            output_origin_y: 0,
        }
    }
}

/// Multi-surface management
///
/// Implements the "Offscreen Surfaces ADM element" from MS-RDPEGFX.
#[derive(Debug, Default)]
pub struct SurfaceManager {
    surfaces: HashMap<u16, Surface>,
    next_surface_id: u16,
}

impl SurfaceManager {
    /// Create a new surface manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a new surface ID
    pub fn allocate_id(&mut self) -> u16 {
        let id = self.next_surface_id;
        self.next_surface_id = self.next_surface_id.wrapping_add(1);
        id
    }

    /// Register a surface
    pub fn insert(&mut self, surface: Surface) {
        self.surfaces.insert(surface.id, surface);
    }

    /// Remove a surface
    pub fn remove(&mut self, surface_id: u16) -> Option<Surface> {
        self.surfaces.remove(&surface_id)
    }

    /// Get a surface by ID
    pub fn get(&self, surface_id: u16) -> Option<&Surface> {
        self.surfaces.get(&surface_id)
    }

    /// Get a mutable surface by ID
    pub fn get_mut(&mut self, surface_id: u16) -> Option<&mut Surface> {
        self.surfaces.get_mut(&surface_id)
    }

    /// Check if a surface exists
    pub fn contains(&self, surface_id: u16) -> bool {
        self.surfaces.contains_key(&surface_id)
    }

    /// Get all surface IDs
    pub fn surface_ids(&self) -> impl Iterator<Item = u16> + '_ {
        self.surfaces.keys().copied()
    }

    /// Clear all surfaces
    pub fn clear(&mut self) {
        self.surfaces.clear();
    }

    /// Number of surfaces
    pub fn len(&self) -> usize {
        self.surfaces.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.surfaces.is_empty()
    }
}

// ============================================================================
// Frame Tracking
// ============================================================================

/// Information about a frame awaiting acknowledgment
///
/// Per MS-RDPEGFX, the server maintains an "Unacknowledged Frames ADM element"
/// which tracks frames sent but not yet acknowledged.
#[derive(Debug, Clone)]
pub struct FrameInfo {
    /// Frame identifier
    pub frame_id: u32,
    /// Frame timestamp
    pub timestamp: Timestamp,
    /// When the frame was sent
    pub sent_at: Instant,
    /// Approximate size in bytes
    pub size_bytes: usize,
}

/// Quality of Experience metrics from client
#[derive(Debug, Clone)]
pub struct QoeMetrics {
    /// Frame ID this relates to
    pub frame_id: u32,
    /// Client timestamp when decode started
    pub timestamp: u32,
    /// Time difference for serial encode (microseconds)
    pub time_diff_se: u16,
    /// Time difference for decode and render (microseconds)
    pub time_diff_dr: u16,
}

/// Frame tracking for flow control
///
/// Implements the "Unacknowledged Frames ADM element" from MS-RDPEGFX.
#[derive(Debug)]
pub struct FrameTracker {
    /// Frames sent but not yet acknowledged
    unacknowledged: HashMap<u32, FrameInfo>,
    /// Last reported client queue depth
    client_queue_depth: u32,
    /// Whether client has suspended acknowledgments
    ack_suspended: bool,
    /// Next frame ID to assign
    next_frame_id: u32,
    /// Maximum frames in flight before backpressure
    max_in_flight: u32,
    /// Total frames sent
    total_sent: u64,
    /// Total frames acknowledged
    total_acked: u64,
}

impl Default for FrameTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameTracker {
    /// Create a new frame tracker
    pub fn new() -> Self {
        Self {
            unacknowledged: HashMap::new(),
            client_queue_depth: 0,
            ack_suspended: false,
            next_frame_id: 0,
            max_in_flight: DEFAULT_MAX_FRAMES_IN_FLIGHT,
            total_sent: 0,
            total_acked: 0,
        }
    }

    /// Set maximum frames in flight
    pub fn set_max_in_flight(&mut self, max: u32) {
        self.max_in_flight = max;
    }

    /// Allocate a new frame ID and track it
    pub fn begin_frame(&mut self, timestamp: Timestamp) -> u32 {
        let frame_id = self.next_frame_id;
        self.next_frame_id = self.next_frame_id.wrapping_add(1);

        self.unacknowledged.insert(
            frame_id,
            FrameInfo {
                frame_id,
                timestamp,
                sent_at: Instant::now(),
                size_bytes: 0,
            },
        );

        self.total_sent += 1;
        frame_id
    }

    /// Update frame size after encoding
    pub fn set_frame_size(&mut self, frame_id: u32, size_bytes: usize) {
        if let Some(info) = self.unacknowledged.get_mut(&frame_id) {
            info.size_bytes = size_bytes;
        }
    }

    /// Handle frame acknowledgment from client
    pub fn acknowledge(&mut self, frame_id: u32, queue_depth: u32) -> Option<FrameInfo> {
        // Update queue depth
        if queue_depth == SUSPEND_FRAME_ACK_QUEUE_DEPTH {
            self.ack_suspended = true;
            self.client_queue_depth = 0;
        } else {
            self.ack_suspended = false;
            self.client_queue_depth = queue_depth;
        }

        // Remove and return the frame info
        let info = self.unacknowledged.remove(&frame_id);
        if info.is_some() {
            self.total_acked += 1;
        }
        info
    }

    /// Number of frames in flight
    #[expect(
        clippy::cast_possible_truncation,
        clippy::as_conversions,
        reason = "frame count will never exceed u32::MAX"
    )]
    pub fn in_flight(&self) -> u32 {
        self.unacknowledged.len() as u32
    }

    /// Check if backpressure should be applied
    pub fn should_backpressure(&self) -> bool {
        !self.ack_suspended && self.in_flight() >= self.max_in_flight
    }

    /// Get client queue depth
    pub fn client_queue_depth(&self) -> u32 {
        self.client_queue_depth
    }

    /// Check if acknowledgments are suspended
    pub fn is_ack_suspended(&self) -> bool {
        self.ack_suspended
    }

    /// Get total frames sent
    pub fn total_sent(&self) -> u64 {
        self.total_sent
    }

    /// Get total frames acknowledged
    pub fn total_acked(&self) -> u64 {
        self.total_acked
    }

    /// Clear all tracking state
    pub fn clear(&mut self) {
        self.unacknowledged.clear();
        self.client_queue_depth = 0;
        self.ack_suspended = false;
    }
}

// ============================================================================
// Capability Negotiation
// ============================================================================

/// Codec capabilities determined from negotiation
#[derive(Debug, Clone, Default)]
pub struct CodecCapabilities {
    /// AVC420 (H.264 4:2:0) is available
    pub avc420: bool,
    /// AVC444 (H.264 4:4:4) is available
    pub avc444: bool,
    /// Small cache mode
    pub small_cache: bool,
    /// Thin client mode
    pub thin_client: bool,
}

impl CodecCapabilities {
    /// Extract codec capabilities from a capability set
    fn from_capability_set(cap: &CapabilitySet) -> Self {
        match cap {
            CapabilitySet::V8 { flags } => Self {
                avc420: false,
                avc444: false,
                small_cache: flags.contains(CapabilitiesV8Flags::SMALL_CACHE),
                thin_client: flags.contains(CapabilitiesV8Flags::THIN_CLIENT),
            },
            CapabilitySet::V8_1 { flags } => Self {
                avc420: flags.contains(CapabilitiesV81Flags::AVC420_ENABLED),
                avc444: false,
                small_cache: flags.contains(CapabilitiesV81Flags::SMALL_CACHE),
                thin_client: flags.contains(CapabilitiesV81Flags::THIN_CLIENT),
            },
            CapabilitySet::V10 { flags } | CapabilitySet::V10_2 { flags } => Self {
                avc420: !flags.contains(CapabilitiesV10Flags::AVC_DISABLED),
                avc444: !flags.contains(CapabilitiesV10Flags::AVC_DISABLED),
                small_cache: flags.contains(CapabilitiesV10Flags::SMALL_CACHE),
                thin_client: false,
            },
            CapabilitySet::V10_1 => Self {
                avc420: true,
                avc444: true,
                small_cache: false,
                thin_client: false,
            },
            CapabilitySet::V10_3 { flags } => Self {
                // V10.3 lacks SMALL_CACHE flag
                avc420: !flags.contains(CapabilitiesV103Flags::AVC_DISABLED),
                avc444: !flags.contains(CapabilitiesV103Flags::AVC_DISABLED),
                small_cache: false,
                thin_client: flags.contains(CapabilitiesV103Flags::AVC_THIN_CLIENT),
            },
            CapabilitySet::V10_4 { flags }
            | CapabilitySet::V10_5 { flags }
            | CapabilitySet::V10_6 { flags }
            | CapabilitySet::V10_6Err { flags } => Self {
                avc420: !flags.contains(CapabilitiesV104Flags::AVC_DISABLED),
                avc444: !flags.contains(CapabilitiesV104Flags::AVC_DISABLED),
                small_cache: flags.contains(CapabilitiesV104Flags::SMALL_CACHE),
                thin_client: flags.contains(CapabilitiesV104Flags::AVC_THIN_CLIENT),
            },
            CapabilitySet::V10_7 { flags } => Self {
                avc420: !flags.contains(CapabilitiesV107Flags::AVC_DISABLED),
                avc444: !flags.contains(CapabilitiesV107Flags::AVC_DISABLED),
                small_cache: flags.contains(CapabilitiesV107Flags::SMALL_CACHE),
                thin_client: flags.contains(CapabilitiesV107Flags::AVC_THIN_CLIENT),
            },
            CapabilitySet::Unknown(_) => Self::default(),
        }
    }
}

/// Priority order for capability negotiation (highest to lowest)
fn capability_priority(cap: &CapabilitySet) -> u32 {
    match cap {
        CapabilitySet::V10_7 { .. } => 12,
        CapabilitySet::V10_6Err { .. } => 11,
        CapabilitySet::V10_6 { .. } => 10,
        CapabilitySet::V10_5 { .. } => 9,
        CapabilitySet::V10_4 { .. } => 8,
        CapabilitySet::V10_3 { .. } => 7,
        CapabilitySet::V10_2 { .. } => 6,
        CapabilitySet::V10_1 => 5,
        CapabilitySet::V10 { .. } => 4,
        CapabilitySet::V8_1 { .. } => 3,
        CapabilitySet::V8 { .. } => 2,
        _ => 0,
    }
}

/// Negotiate the best capability set between client and server
///
/// Per MS-RDPEGFX section 3.3.5.1.2, the server MUST select one of the
/// capability sets **advertised by the client**. We prioritize based on
/// the server's preference order, but return the client's capability set
/// (with the client's flags).
fn negotiate_capabilities(client_caps: &[CapabilitySet], server_caps: &[CapabilitySet]) -> Option<CapabilitySet> {
    // Sort server capabilities by priority (highest first)
    let mut server_sorted: Vec<_> = server_caps.iter().collect();
    server_sorted.sort_by_key(|cap| core::cmp::Reverse(capability_priority(cap)));

    // Find highest priority server cap that client also supports
    // Return the CLIENT's capability set (with client's flags), not server's
    for server_cap in server_sorted {
        for client_cap in client_caps {
            if core::mem::discriminant(client_cap) == core::mem::discriminant(server_cap) {
                return Some(client_cap.clone());
            }
        }
    }

    None
}

// ============================================================================
// Handler Trait
// ============================================================================

/// Handler trait for server-side EGFX events
///
/// Implement this trait to receive callbacks when the EGFX channel state changes
/// or when client messages are received.
pub trait GraphicsPipelineHandler: Send {
    /// Called when the client advertises its capabilities
    ///
    /// This is informational - the server will automatically negotiate
    /// based on [`preferred_capabilities()`](Self::preferred_capabilities).
    fn capabilities_advertise(&mut self, pdu: &CapabilitiesAdvertisePdu);

    /// Called when the EGFX channel is ready to send frames
    ///
    /// At this point, capability negotiation is complete.
    /// The handler should create surfaces and start sending frames.
    fn on_ready(&mut self, negotiated: &CapabilitySet);

    /// Called when a frame has been acknowledged by the client
    ///
    /// # Arguments
    ///
    /// * `frame_id` - The acknowledged frame
    /// * `queue_depth` - Client's reported queue depth (bytes buffered)
    fn on_frame_ack(&mut self, _frame_id: u32, _queue_depth: u32) {}

    /// Called when QoE metrics are received from client (V10+)
    fn on_qoe_metrics(&mut self, _metrics: QoeMetrics) {}

    /// Called when a surface is created
    fn on_surface_created(&mut self, _surface: &Surface) {}

    /// Called when a surface is deleted
    fn on_surface_deleted(&mut self, _surface_id: u16) {}

    /// Called when the EGFX channel is closed
    fn on_close(&mut self) {}

    /// Returns the server's preferred capabilities
    ///
    /// Override this to customize codec support. The default enables
    /// AVC420/AVC444 with V10.7 and V8.1 as fallback.
    fn preferred_capabilities(&self) -> Vec<CapabilitySet> {
        vec![
            // Prefer V10.7 with AVC enabled
            CapabilitySet::V10_7 {
                flags: CapabilitiesV107Flags::SMALL_CACHE,
            },
            // V10 fallback
            CapabilitySet::V10 {
                flags: CapabilitiesV10Flags::SMALL_CACHE,
            },
            // V8.1 with AVC420
            CapabilitySet::V8_1 {
                flags: CapabilitiesV81Flags::AVC420_ENABLED | CapabilitiesV81Flags::SMALL_CACHE,
            },
            // V8 basic fallback
            CapabilitySet::V8 {
                flags: CapabilitiesV8Flags::SMALL_CACHE,
            },
        ]
    }

    /// Returns the maximum frames in flight before backpressure
    fn max_frames_in_flight(&self) -> u32 {
        DEFAULT_MAX_FRAMES_IN_FLIGHT
    }

    /// Called when client offers to import cached bitmaps
    ///
    /// Return the list of cache slot IDs to accept.
    /// Default rejects all (returns empty).
    fn on_cache_import_offer(&mut self, _offer: &CacheImportOfferPdu) -> Vec<u16> {
        vec![]
    }
}

// ============================================================================
// Server State Machine
// ============================================================================

/// Server state machine states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServerState {
    /// Waiting for client CapabilitiesAdvertise
    WaitingForCapabilities,
    /// Channel is ready, can send frames
    Ready,
    /// Performing a resize operation
    Resizing,
    /// Channel has been closed
    Closed,
}

// ============================================================================
// Graphics Pipeline Server
// ============================================================================

/// Server for the Graphics Pipeline Virtual Channel (EGFX)
///
/// This server handles capability negotiation, surface management,
/// and H.264 frame transmission to RDP clients per MS-RDPEGFX specification.
pub struct GraphicsPipelineServer {
    handler: Box<dyn GraphicsPipelineHandler>,

    // State management
    state: ServerState,
    negotiated_caps: Option<CapabilitySet>,
    codec_caps: CodecCapabilities,

    // Surface management (Offscreen Surfaces ADM element)
    surfaces: SurfaceManager,

    // Frame tracking (Unacknowledged Frames ADM element)
    frames: FrameTracker,

    // Graphics output buffer dimensions
    output_width: u16,
    output_height: u16,

    // Whether ResetGraphics has been sent
    // Per MS-RDPEGFX, must be sent before any CreateSurface
    reset_graphics_sent: bool,

    // Output queue for PDUs that need to be sent
    output_queue: VecDeque<GfxPdu>,

    // DVC channel ID assigned by DRDYNVC
    // Set when start() is called, needed for encode_dvc_messages()
    channel_id: Option<u32>,

    // ZGFX compression
    zgfx_compressor: Compressor,
    compression_mode: CompressionMode,
}

impl GraphicsPipelineServer {
    /// Create a new GraphicsPipelineServer
    pub fn new(handler: Box<dyn GraphicsPipelineHandler>) -> Self {
        // Use Never mode - H.264 video is already compressed by the codec
        // Attempting ZGFX compression on H.264 provides no benefit and wastes CPU
        //
        // Never mode:
        // - Just wraps in ZGFX segment structure (2-byte overhead)
        // - <1µs processing time per PDU
        // - No hash table maintenance
        // - Production-ready and stable
        //
        // Note: Auto/Always modes available via with_compression() if needed
        Self::with_compression(handler, CompressionMode::Never)
    }

    /// Create a new GraphicsPipelineServer with specified compression mode
    ///
    /// # Arguments
    ///
    /// * `handler` - Handler for server callbacks
    /// * `compression_mode` - ZGFX compression mode (Never/Auto/Always)
    pub fn with_compression(handler: Box<dyn GraphicsPipelineHandler>, compression_mode: CompressionMode) -> Self {
        let max_frames = handler.max_frames_in_flight();
        let mut frames = FrameTracker::new();
        frames.set_max_in_flight(max_frames);

        Self {
            handler,
            state: ServerState::WaitingForCapabilities,
            negotiated_caps: None,
            codec_caps: CodecCapabilities::default(),
            surfaces: SurfaceManager::new(),
            frames,
            output_width: 0,
            output_height: 0,
            reset_graphics_sent: false,
            output_queue: VecDeque::new(),
            channel_id: None,
            zgfx_compressor: Compressor::new(),
            compression_mode,
        }
    }

    /// Set ZGFX compression mode
    ///
    /// This can be called at any time to change compression behavior.
    pub fn set_compression_mode(&mut self, mode: CompressionMode) {
        self.compression_mode = mode;
        debug!("ZGFX compression mode set to: {:?}", mode);
    }

    /// Get current compression mode
    pub fn compression_mode(&self) -> CompressionMode {
        self.compression_mode
    }

    /// Set the desktop output dimensions for ResetGraphics
    ///
    /// Call this BEFORE create_surface() to control the desktop size announced
    /// to the client, which may differ from the surface size (for 16-pixel alignment).
    ///
    /// # Arguments
    ///
    /// * `width` - Desktop width (what client sees)
    /// * `height` - Desktop height (what client sees)
    pub fn set_output_dimensions(&mut self, width: u16, height: u16) {
        self.output_width = width;
        self.output_height = height;
        debug!(width, height, "Output dimensions configured for ResetGraphics");
    }

    /// Get the DVC channel ID assigned to this EGFX channel
    ///
    /// Returns `None` if the channel hasn't been started yet.
    /// Use this to encode DVC messages for proactive frame sending.
    #[must_use]
    pub fn channel_id(&self) -> Option<u32> {
        self.channel_id
    }

    // ========================================================================
    // State Queries
    // ========================================================================

    /// Check if the server is ready to send frames
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.state == ServerState::Ready
    }

    /// Get the negotiated capability set
    #[must_use]
    pub fn negotiated_capabilities(&self) -> Option<&CapabilitySet> {
        self.negotiated_caps.as_ref()
    }

    /// Get codec capabilities determined from negotiation
    #[must_use]
    pub fn codec_capabilities(&self) -> &CodecCapabilities {
        &self.codec_caps
    }

    /// Check if AVC420 (H.264 4:2:0) is available
    #[must_use]
    pub fn supports_avc420(&self) -> bool {
        self.codec_caps.avc420
    }

    /// Check if AVC444 (H.264 4:4:4) is available
    #[must_use]
    pub fn supports_avc444(&self) -> bool {
        self.codec_caps.avc444
    }

    /// Get the graphics output buffer dimensions
    #[must_use]
    pub fn output_dimensions(&self) -> (u16, u16) {
        (self.output_width, self.output_height)
    }

    // ========================================================================
    // Surface Management
    // ========================================================================

    /// Create a new surface
    ///
    /// Queues CreateSurface PDU and returns the surface ID.
    /// Returns `None` if not ready.
    pub fn create_surface(&mut self, width: u16, height: u16) -> Option<u16> {
        self.create_surface_with_format(width, height, PixelFormat::XRgb)
    }

    /// Create a new surface with specific pixel format
    pub fn create_surface_with_format(&mut self, width: u16, height: u16, pixel_format: PixelFormat) -> Option<u16> {
        if self.state != ServerState::Ready && self.state != ServerState::Resizing {
            return None;
        }

        // Per MS-RDPEGFX, ResetGraphics MUST be sent before any CreateSurface
        // Send it automatically on first surface creation if not already sent
        // CRITICAL: Use output_width/output_height if already set (from manual call)
        // Otherwise use surface dimensions
        if !self.reset_graphics_sent {
            let desktop_width = if self.output_width > 0 { self.output_width } else { width };
            let desktop_height = if self.output_height > 0 { self.output_height } else { height };

            self.output_queue.push_back(GfxPdu::ResetGraphics(ResetGraphicsPdu {
                width: u32::from(desktop_width),
                height: u32::from(desktop_height),
                monitors: Vec::new(),
            }));

            self.output_width = desktop_width;
            self.output_height = desktop_height;
            self.reset_graphics_sent = true;
            debug!(desktop_width, desktop_height, surface_width=width, surface_height=height,
                   "Sent ResetGraphics before first surface");
        }

        let surface_id = self.surfaces.allocate_id();
        let surface = Surface::new(surface_id, width, height, pixel_format);

        // Queue CreateSurface PDU
        self.output_queue.push_back(GfxPdu::CreateSurface(CreateSurfacePdu {
            surface_id,
            width,
            height,
            pixel_format,
        }));

        self.handler.on_surface_created(&surface);
        self.surfaces.insert(surface);

        debug!(surface_id, width, height, ?pixel_format, "Created surface");
        Some(surface_id)
    }

    /// Delete a surface
    ///
    /// Queues DeleteSurface PDU. Returns `false` if surface doesn't exist.
    pub fn delete_surface(&mut self, surface_id: u16) -> bool {
        if self.surfaces.remove(surface_id).is_none() {
            return false;
        }

        // Queue DeleteSurface PDU
        self.output_queue
            .push_back(GfxPdu::DeleteSurface(DeleteSurfacePdu { surface_id }));

        self.handler.on_surface_deleted(surface_id);
        debug!(surface_id, "Deleted surface");
        true
    }

    /// Map a surface to the graphics output buffer
    pub fn map_surface_to_output(&mut self, surface_id: u16, origin_x: u32, origin_y: u32) -> bool {
        let Some(surface) = self.surfaces.get_mut(surface_id) else {
            return false;
        };

        surface.is_mapped = true;
        surface.output_origin_x = origin_x;
        surface.output_origin_y = origin_y;

        self.output_queue
            .push_back(GfxPdu::MapSurfaceToOutput(MapSurfaceToOutputPdu {
                surface_id,
                output_origin_x: origin_x,
                output_origin_y: origin_y,
            }));

        debug!(surface_id, origin_x, origin_y, "Mapped surface to output");
        true
    }

    /// Get a surface by ID
    #[must_use]
    pub fn get_surface(&self, surface_id: u16) -> Option<&Surface> {
        self.surfaces.get(surface_id)
    }

    /// Get all surface IDs
    pub fn surface_ids(&self) -> impl Iterator<Item = u16> + '_ {
        self.surfaces.surface_ids()
    }

    // ========================================================================
    // Resize Handling
    // ========================================================================

    /// Resize the graphics output buffer
    ///
    /// This initiates a resize sequence:
    /// 1. Sends ResetGraphics with new dimensions
    /// 2. Deletes existing surfaces
    /// 3. Transitions to Ready state
    ///
    /// After calling this, create new surfaces for the new dimensions.
    pub fn resize(&mut self, width: u16, height: u16) {
        self.resize_with_monitors(width, height, Vec::new());
    }

    /// Resize with explicit monitor configuration
    pub fn resize_with_monitors(&mut self, width: u16, height: u16, monitors: Vec<Monitor>) {
        if self.state != ServerState::Ready {
            debug!("Cannot resize: not in Ready state");
            return;
        }

        debug!(width, height, monitors = monitors.len(), "Initiating resize");

        self.state = ServerState::Resizing;
        self.output_width = width;
        self.output_height = height;

        // Delete all existing surfaces
        let surface_ids: Vec<_> = self.surfaces.surface_ids().collect();
        for id in surface_ids {
            self.delete_surface(id);
        }

        // Clear frame tracking
        self.frames.clear();

        // Send ResetGraphics
        self.output_queue.push_back(GfxPdu::ResetGraphics(ResetGraphicsPdu {
            width: u32::from(width),
            height: u32::from(height),
            monitors,
        }));

        // Mark that ResetGraphics has been sent
        self.reset_graphics_sent = true;

        // Return to Ready state
        self.state = ServerState::Ready;
    }

    // ========================================================================
    // Flow Control
    // ========================================================================

    /// Check if backpressure should be applied
    ///
    /// Returns `true` if too many frames are in flight and the caller
    /// should drop or delay new frames.
    #[must_use]
    pub fn should_backpressure(&self) -> bool {
        self.frames.should_backpressure()
    }

    /// Get the number of frames currently in flight (awaiting ACK)
    #[must_use]
    pub fn frames_in_flight(&self) -> u32 {
        self.frames.in_flight()
    }

    /// Get the last reported client queue depth
    #[must_use]
    pub fn client_queue_depth(&self) -> u32 {
        self.frames.client_queue_depth()
    }

    /// Set the maximum frames in flight before backpressure
    pub fn set_max_frames_in_flight(&mut self, max: u32) {
        self.frames.set_max_in_flight(max);
    }

    // ========================================================================
    // Frame Sending
    // ========================================================================

    /// Convert timestamp in milliseconds to Timestamp struct
    #[expect(
        clippy::as_conversions,
        reason = "arithmetic results bounded and fit in target types"
    )]
    fn make_timestamp(timestamp_ms: u32) -> Timestamp {
        Timestamp {
            milliseconds: (timestamp_ms % 1000) as u16,
            seconds: ((timestamp_ms / 1000) % 60) as u8,
            minutes: ((timestamp_ms / 60000) % 60) as u8,
            hours: ((timestamp_ms / 3600000) % 24) as u16,
        }
    }

    /// Compute bounding rectangle from regions
    fn compute_dest_rect(regions: &[Avc420Region], default_width: u16, default_height: u16) -> InclusiveRectangle {
        if let Some(first) = regions.first() {
            let mut left = first.left;
            let mut top = first.top;
            let mut right = first.right;
            let mut bottom = first.bottom;

            for r in regions.iter().skip(1) {
                left = left.min(r.left);
                top = top.min(r.top);
                right = right.max(r.right);
                bottom = bottom.max(r.bottom);
            }

            InclusiveRectangle {
                left,
                top,
                right,
                bottom,
            }
        } else {
            InclusiveRectangle {
                left: 0,
                top: 0,
                right: default_width.saturating_sub(1),
                bottom: default_height.saturating_sub(1),
            }
        }
    }

    /// Queue an H.264 AVC420 frame for transmission
    ///
    /// # Arguments
    ///
    /// * `surface_id` - Target surface
    /// * `h264_data` - H.264 encoded data in AVC format (use `annex_b_to_avc` if needed)
    /// * `regions` - List of regions describing the frame
    /// * `timestamp_ms` - Frame timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// `Some(frame_id)` if the frame was queued, `None` if backpressure is active,
    /// server is not ready, or AVC420 is not supported.
    pub fn send_avc420_frame(
        &mut self,
        surface_id: u16,
        h264_data: &[u8],
        regions: &[Avc420Region],
        timestamp_ms: u32,
    ) -> Option<u32> {
        if !self.is_ready() {
            debug!("EGFX not ready, dropping frame");
            return None;
        }

        if !self.supports_avc420() {
            debug!("AVC420 not supported, dropping frame");
            return None;
        }

        if self.should_backpressure() {
            trace!(frames_in_flight = self.frames.in_flight(), "EGFX backpressure active");
            return None;
        }

        let Some(surface) = self.surfaces.get(surface_id) else {
            debug!(surface_id, "Surface not found, dropping frame");
            return None;
        };

        let timestamp = Self::make_timestamp(timestamp_ms);
        let frame_id = self.frames.begin_frame(timestamp);

        // Log region details for debugging
        for (i, region) in regions.iter().enumerate() {
            trace!(
                "Region[{}]: left={}, top={}, right={}, bottom={}, qp={}, quality={}",
                i, region.left, region.top, region.right, region.bottom,
                region.quantization_parameter, region.quality
            );
        }

        // Build the bitmap data
        let bitmap_data = encode_avc420_bitmap_stream(regions, h264_data);

        // Determine destination rectangle
        let dest_rect = Self::compute_dest_rect(regions, surface.width, surface.height);

        trace!(
            "DestRect: left={}, top={}, right={}, bottom={} | BitmapStream: {} bytes | H264: {} bytes",
            dest_rect.left, dest_rect.top, dest_rect.right, dest_rect.bottom,
            bitmap_data.len(), h264_data.len()
        );

        // Queue the frame PDUs
        self.output_queue
            .push_back(GfxPdu::StartFrame(StartFramePdu { timestamp, frame_id }));

        self.output_queue.push_back(GfxPdu::WireToSurface1(WireToSurface1Pdu {
            surface_id,
            codec_id: Codec1Type::Avc420,
            pixel_format: surface.pixel_format,
            destination_rectangle: dest_rect,
            bitmap_data,
        }));

        self.output_queue.push_back(GfxPdu::EndFrame(EndFramePdu { frame_id }));

        trace!(frame_id, surface_id, "Queued AVC420 frame");
        Some(frame_id)
    }

    /// Queue an H.264 AVC444 frame for transmission
    ///
    /// AVC444 uses two streams: one for luma (Y) and one for chroma (UV).
    /// If only luma data is provided, set `chroma_data` to `None`.
    ///
    /// # Arguments
    ///
    /// * `surface_id` - Target surface
    /// * `luma_data` - H.264 encoded luma (Y) data in AVC format
    /// * `luma_regions` - Regions for luma stream
    /// * `chroma_data` - Optional H.264 encoded chroma (UV) data
    /// * `chroma_regions` - Regions for chroma stream (required if chroma_data provided)
    /// * `timestamp_ms` - Frame timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// `Some(frame_id)` if the frame was queued, `None` if not supported or backpressured.
    pub fn send_avc444_frame(
        &mut self,
        surface_id: u16,
        luma_data: &[u8],
        luma_regions: &[Avc420Region],
        chroma_data: Option<&[u8]>,
        chroma_regions: Option<&[Avc420Region]>,
        timestamp_ms: u32,
    ) -> Option<u32> {
        if !self.is_ready() {
            debug!("EGFX not ready, dropping frame");
            return None;
        }

        if !self.supports_avc444() {
            debug!("AVC444 not supported, dropping frame");
            return None;
        }

        if self.should_backpressure() {
            trace!(frames_in_flight = self.frames.in_flight(), "EGFX backpressure active");
            return None;
        }

        let Some(surface) = self.surfaces.get(surface_id) else {
            debug!(surface_id, "Surface not found, dropping frame");
            return None;
        };

        let timestamp = Self::make_timestamp(timestamp_ms);
        let frame_id = self.frames.begin_frame(timestamp);

        // Build luma stream
        let luma_rectangles: Vec<_> = luma_regions.iter().map(Avc420Region::to_rectangle).collect();
        let luma_quant_vals: Vec<_> = luma_regions.iter().map(Avc420Region::to_quant_quality).collect();

        let stream1 = Avc420BitmapStream {
            rectangles: luma_rectangles,
            quant_qual_vals: luma_quant_vals,
            data: luma_data,
        };

        // Build chroma stream if provided
        let (encoding, stream2) = if let (Some(chroma), Some(chroma_regs)) = (chroma_data, chroma_regions) {
            let chroma_rectangles: Vec<_> = chroma_regs.iter().map(Avc420Region::to_rectangle).collect();
            let chroma_quant_vals: Vec<_> = chroma_regs.iter().map(Avc420Region::to_quant_quality).collect();

            (
                Encoding::LUMA_AND_CHROMA,
                Some(Avc420BitmapStream {
                    rectangles: chroma_rectangles,
                    quant_qual_vals: chroma_quant_vals,
                    data: chroma,
                }),
            )
        } else {
            (Encoding::LUMA, None)
        };

        let avc444_stream = Avc444BitmapStream {
            encoding,
            stream1,
            stream2,
        };

        // Encode the AVC444 stream
        let bitmap_data = encode_avc444_bitmap_stream(&avc444_stream);

        // Determine destination rectangle
        let dest_rect = Self::compute_dest_rect(luma_regions, surface.width, surface.height);

        // Queue the frame PDUs
        self.output_queue
            .push_back(GfxPdu::StartFrame(StartFramePdu { timestamp, frame_id }));

        self.output_queue.push_back(GfxPdu::WireToSurface1(WireToSurface1Pdu {
            surface_id,
            codec_id: Codec1Type::Avc444,
            pixel_format: surface.pixel_format,
            destination_rectangle: dest_rect,
            bitmap_data,
        }));

        self.output_queue.push_back(GfxPdu::EndFrame(EndFramePdu { frame_id }));

        trace!(frame_id, surface_id, "Queued AVC444 frame");
        Some(frame_id)
    }

    // ========================================================================
    // Output Management
    // ========================================================================

    /// Drain the output queue and return PDUs to send
    ///
    /// Call this method to get pending PDUs that need to be sent to the client.
    ///
    /// # ZGFX Wrapping and Compression
    ///
    /// Each GfxPdu is:
    /// 1. Encoded to bytes
    /// 2. Optionally ZGFX-compressed (based on compression_mode)
    /// 3. Wrapped in ZGFX segment structure
    ///
    /// This ensures Windows clients can properly decode the PDUs.
    #[expect(clippy::as_conversions, reason = "Box<T> to Box<dyn Trait> coercion")]
    pub fn drain_output(&mut self) -> Vec<DvcMessage> {
        let compression_mode = self.compression_mode;

        let messages: Vec<DvcMessage> = self.output_queue
            .drain(..)
            .map(|pdu| {
                // Get PDU name for logging
                let pdu_name = match &pdu {
                    GfxPdu::CapabilitiesConfirm(caps) => {
                        debug!("Draining CapabilitiesConfirm: {:?} (ZGFX mode: {:?})", caps.0, compression_mode);
                        "CapabilitiesConfirm"
                    }
                    GfxPdu::ResetGraphics(_) => {
                        debug!("Draining ResetGraphics PDU (ZGFX mode: {:?})", compression_mode);
                        "ResetGraphics"
                    }
                    GfxPdu::CreateSurface(p) => {
                        debug!("Draining CreateSurface: id={}, {}x{} (ZGFX mode: {:?})", p.surface_id, p.width, p.height, compression_mode);
                        "CreateSurface"
                    }
                    GfxPdu::MapSurfaceToOutput(p) => {
                        debug!("Draining MapSurfaceToOutput: id={} (ZGFX mode: {:?})", p.surface_id, compression_mode);
                        "MapSurfaceToOutput"
                    }
                    GfxPdu::StartFrame(p) => {
                        trace!("Draining StartFrame: id={} (ZGFX mode: {:?})", p.frame_id, compression_mode);
                        "StartFrame"
                    }
                    GfxPdu::WireToSurface1(_) => {
                        trace!("Draining WireToSurface1 (ZGFX mode: {:?})", compression_mode);
                        "WireToSurface1"
                    }
                    GfxPdu::EndFrame(p) => {
                        trace!("Draining EndFrame: id={} (ZGFX mode: {:?})", p.frame_id, compression_mode);
                        "EndFrame"
                    }
                    _ => {
                        trace!("Draining other GfxPdu (ZGFX mode: {:?})", compression_mode);
                        "OtherGfxPdu"
                    }
                };

                // Encode GfxPdu to bytes
                let gfx_size = pdu.size();
                let mut gfx_bytes = vec![0u8; gfx_size];
                let mut gfx_cursor = WriteCursor::new(&mut gfx_bytes);
                pdu.encode(&mut gfx_cursor).expect("GfxPdu encode should not fail");

                // Compress and wrap with ZGFX (with performance timing)
                debug!("🗜️  ZGFX input: {} bytes, mode: {:?}, PDU: {}", gfx_size, compression_mode, pdu_name);
                let start = Instant::now();
                let zgfx_wrapped = zgfx::compress_and_wrap_egfx(
                    &gfx_bytes,
                    &mut self.zgfx_compressor,
                    compression_mode,
                ).expect("ZGFX compression should not fail");
                let duration = start.elapsed();

                // Log compression effectiveness and performance
                let ratio = gfx_size as f64 / zgfx_wrapped.len() as f64;
                let compressed = zgfx_wrapped.len() < gfx_size + 2; // +2 for wrapper overhead
                debug!(
                    "🗜️  ZGFX output: {} bytes (ratio: {:.2}x, {}, time: {:?})",
                    zgfx_wrapped.len(),
                    ratio,
                    if compressed { "compressed" } else { "uncompressed" },
                    duration
                );

                if gfx_size > 1000 {
                    trace!(
                        "{}: {} bytes → {} bytes (ratio: {:.2}x)",
                        pdu_name,
                        gfx_size,
                        zgfx_wrapped.len(),
                        ratio
                    );
                }

                Box::new(ZgfxWrappedBytes::new(zgfx_wrapped, pdu_name)) as DvcMessage
            })
            .collect();

        if !messages.is_empty() {
            debug!(
                "drain_output returning {} ZGFX-wrapped messages (mode: {:?})",
                messages.len(),
                compression_mode
            );
        }

        messages
    }

    /// Check if there are pending PDUs to send
    #[must_use]
    pub fn has_pending_output(&self) -> bool {
        !self.output_queue.is_empty()
    }

    // ========================================================================
    // Internal Message Handlers
    // ========================================================================

    /// Handle capability negotiation
    fn handle_capabilities_advertise(&mut self, pdu: CapabilitiesAdvertisePdu) {
        debug!(?pdu, "Received CapabilitiesAdvertise");

        // Notify handler
        self.handler.capabilities_advertise(&pdu);

        // Get server's preferred capabilities
        let server_caps = self.handler.preferred_capabilities();

        // Negotiate best match
        let negotiated = negotiate_capabilities(&pdu.0, &server_caps).unwrap_or_else(|| {
            // Fallback to V8.1 with AVC420
            warn!("No matching capabilities, falling back to V8.1");
            CapabilitySet::V8_1 {
                flags: CapabilitiesV81Flags::AVC420_ENABLED,
            }
        });

        debug!(?negotiated, "Negotiated capabilities");

        // Extract codec capabilities
        self.codec_caps = CodecCapabilities::from_capability_set(&negotiated);
        self.negotiated_caps = Some(negotiated.clone());

        // Queue CapabilitiesConfirm
        self.output_queue
            .push_back(GfxPdu::CapabilitiesConfirm(CapabilitiesConfirmPdu(negotiated.clone())));

        debug!(
            "Queued CapabilitiesConfirm for {:?} (output_queue size: {})",
            negotiated,
            self.output_queue.len()
        );

        // Transition to ready state
        self.state = ServerState::Ready;

        // Notify handler
        self.handler.on_ready(&negotiated);

        debug!(
            avc420 = self.codec_caps.avc420,
            avc444 = self.codec_caps.avc444,
            "EGFX server ready"
        );
    }

    /// Handle frame acknowledgment
    fn handle_frame_acknowledge(&mut self, pdu: FrameAcknowledgePdu) {
        trace!(?pdu, "Received FrameAcknowledge");

        // Convert QueueDepth enum to u32 for tracking
        let queue_depth_u32 = pdu.queue_depth.to_u32();

        if let Some(info) = self.frames.acknowledge(pdu.frame_id, queue_depth_u32) {
            let latency = info.sent_at.elapsed();
            trace!(frame_id = pdu.frame_id, ?latency, "Frame acknowledged");
        }

        self.handler.on_frame_ack(pdu.frame_id, queue_depth_u32);
    }

    /// Handle QoE frame acknowledgment
    fn handle_qoe_frame_acknowledge(&mut self, pdu: QoeFrameAcknowledgePdu) {
        trace!(?pdu, "Received QoeFrameAcknowledge");

        let metrics = QoeMetrics {
            frame_id: pdu.frame_id,
            timestamp: pdu.timestamp,
            time_diff_se: pdu.time_diff_se,
            time_diff_dr: pdu.time_diff_dr,
        };

        self.handler.on_qoe_metrics(metrics);
    }

    /// Handle cache import offer
    fn handle_cache_import_offer(&mut self, pdu: CacheImportOfferPdu) {
        debug!(entries = pdu.cache_entries.len(), "Received CacheImportOffer");

        // Ask handler which entries to accept
        let accepted = self.handler.on_cache_import_offer(&pdu);

        // Send reply
        self.output_queue
            .push_back(GfxPdu::CacheImportReply(CacheImportReplyPdu { cache_slots: accepted }));
    }
}

impl_as_any!(GraphicsPipelineServer);

impl DvcProcessor for GraphicsPipelineServer {
    fn channel_name(&self) -> &str {
        CHANNEL_NAME
    }

    fn start(&mut self, channel_id: u32) -> PduResult<Vec<DvcMessage>> {
        // Store channel_id for later use by proactive frame sending
        self.channel_id = Some(channel_id);
        debug!(channel_id, "EGFX channel started");
        // Server doesn't send anything at start - waits for client CapabilitiesAdvertise
        Ok(vec![])
    }

    fn close(&mut self, _channel_id: u32) {
        debug!("EGFX channel closed");
        self.state = ServerState::Closed;
        self.reset_graphics_sent = false;
        self.handler.on_close();
    }

    fn process(&mut self, _channel_id: u32, payload: &[u8]) -> PduResult<Vec<DvcMessage>> {
        let pdu = decode(payload).map_err(|e| decode_err!(e))?;

        match pdu {
            GfxPdu::CapabilitiesAdvertise(pdu) => {
                self.handle_capabilities_advertise(pdu);
            }
            GfxPdu::FrameAcknowledge(pdu) => {
                self.handle_frame_acknowledge(pdu);
            }
            GfxPdu::QoeFrameAcknowledge(pdu) => {
                self.handle_qoe_frame_acknowledge(pdu);
            }
            GfxPdu::CacheImportOffer(pdu) => {
                self.handle_cache_import_offer(pdu);
            }
            _ => {
                warn!(?pdu, "Unhandled client GFX PDU");
            }
        }

        // Return any queued output
        Ok(self.drain_output())
    }
}

impl DvcServerProcessor for GraphicsPipelineServer {}

// ============================================================================
// AVC444 Encoding Helper
// ============================================================================

/// Encode an AVC444 bitmap stream to bytes
fn encode_avc444_bitmap_stream(stream: &Avc444BitmapStream<'_>) -> Vec<u8> {
    use ironrdp_pdu::{Encode as _, WriteCursor};

    let size = stream.size();
    let mut buf = vec![0u8; size];
    let mut cursor = WriteCursor::new(&mut buf);

    stream
        .encode(&mut cursor)
        .expect("encode_avc444_bitmap_stream: encoding failed");

    buf
}
