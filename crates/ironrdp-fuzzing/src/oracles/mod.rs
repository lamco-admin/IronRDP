//! Oracles.
//!
//! Oracles take a test case and determine whether we have a bug. For example,
//! one of the simplest oracles is to take a RDP PDU as our input test case,
//! encode and decode it, and (implicitly) check that no assertions
//! failed or segfaults happened. A more complicated oracle might compare the
//! result of two different implementations for the same thing, and
//! make sure that the two executions are observably identical (differential fuzzing).
//!
//! When an oracle finds a bug, it should report it to the fuzzing engine by
//! panicking.

use crate::generators::BitmapInput;

pub fn pdu_decode(data: &[u8]) {
    use ironrdp_core::decode;
    use ironrdp_pdu::mcs::{ConnectInitial, ConnectResponse, McsMessage};
    use ironrdp_pdu::nego::{ConnectionConfirm, ConnectionRequest};
    use ironrdp_pdu::rdp::{ClientInfoPdu, capability_sets, headers, server_error_info, server_license, vc};
    use ironrdp_pdu::x224::X224;
    use ironrdp_pdu::{bitmap, codecs, fast_path, gcc, input, pcb, surface_commands};

    let _ = decode::<X224<ConnectionRequest>>(data);
    let _ = decode::<X224<ConnectionConfirm>>(data);
    let _ = decode::<X224<McsMessage<'_>>>(data);
    let _ = decode::<ConnectInitial>(data);
    let _ = decode::<ConnectResponse>(data);
    let _ = decode::<ClientInfoPdu>(data);
    let _ = decode::<capability_sets::CapabilitySet>(data);
    let _ = decode::<headers::ShareControlHeader>(data);
    let _ = decode::<pcb::PreconnectionBlob>(data);
    let _ = decode::<server_error_info::ServerSetErrorInfoPdu>(data);

    let _ = decode::<gcc::ClientGccBlocks>(data);
    let _ = decode::<gcc::ServerGccBlocks>(data);
    let _ = decode::<gcc::ClientClusterData>(data);
    let _ = decode::<gcc::ConferenceCreateRequest>(data);
    let _ = decode::<gcc::ConferenceCreateResponse>(data);

    let _ = decode::<server_license::LicensePdu>(data);

    let _ = decode::<vc::ChannelPduHeader>(data);

    let _ = decode::<fast_path::FastPathHeader>(data);
    let _ = decode::<fast_path::FastPathUpdatePdu<'_>>(data);
    let _ = fast_path::FastPathUpdate::decode_with_code(data, fast_path::UpdateCode::Orders);
    let _ = fast_path::FastPathUpdate::decode_with_code(data, fast_path::UpdateCode::Bitmap);
    let _ = fast_path::FastPathUpdate::decode_with_code(data, fast_path::UpdateCode::Palette);
    let _ = fast_path::FastPathUpdate::decode_with_code(data, fast_path::UpdateCode::Synchronize);
    let _ = fast_path::FastPathUpdate::decode_with_code(data, fast_path::UpdateCode::SurfaceCommands);
    let _ = fast_path::FastPathUpdate::decode_with_code(data, fast_path::UpdateCode::HiddenPointer);
    let _ = fast_path::FastPathUpdate::decode_with_code(data, fast_path::UpdateCode::DefaultPointer);
    let _ = fast_path::FastPathUpdate::decode_with_code(data, fast_path::UpdateCode::PositionPointer);
    let _ = fast_path::FastPathUpdate::decode_with_code(data, fast_path::UpdateCode::ColorPointer);
    let _ = fast_path::FastPathUpdate::decode_with_code(data, fast_path::UpdateCode::CachedPointer);
    let _ = fast_path::FastPathUpdate::decode_with_code(data, fast_path::UpdateCode::NewPointer);
    let _ = fast_path::FastPathUpdate::decode_with_code(data, fast_path::UpdateCode::LargePointer);

    let _ = decode::<surface_commands::SurfaceCommand<'_>>(data);
    let _ = decode::<surface_commands::SurfaceBitsPdu<'_>>(data);
    let _ = decode::<surface_commands::FrameMarkerPdu>(data);
    let _ = decode::<surface_commands::ExtendedBitmapDataPdu<'_>>(data);
    let _ = decode::<surface_commands::BitmapDataHeader>(data);

    let _ = decode::<codecs::rfx::Block<'_>>(data);

    let _ = decode::<input::InputEventPdu>(data);
    let _ = decode::<input::InputEvent>(data);

    let _ = decode::<bitmap::rdp6::BitmapStream<'_>>(data);

    let _ = decode::<ironrdp_cliprdr::pdu::ClipboardPdu<'_>>(data);
    let _ = decode::<ironrdp_cliprdr::pdu::PackedFileList>(data);
    let _ = decode::<ironrdp_cliprdr::pdu::FileContentsRequest>(data);
    let _ = decode::<ironrdp_cliprdr::pdu::FileContentsResponse<'_>>(data);

    let _ = decode::<ironrdp_rdpdr::pdu::RdpdrPdu>(data);

    let _ = decode::<ironrdp_displaycontrol::pdu::DisplayControlPdu>(data);

    let _ = decode::<ironrdp_rdpsnd::pdu::ServerAudioOutputPdu<'_>>(data);
    let _ = decode::<ironrdp_rdpsnd::pdu::ClientAudioOutputPdu>(data);
}

pub fn rle_decompress_bitmap(input: BitmapInput<'_>) {
    let mut out = Vec::new();

    let _ = ironrdp_graphics::rle::decompress_24_bpp(input.src, &mut out, input.width.into(), input.height.into());
    let _ = ironrdp_graphics::rle::decompress_16_bpp(input.src, &mut out, input.width.into(), input.height.into());
    let _ = ironrdp_graphics::rle::decompress_15_bpp(input.src, &mut out, input.width.into(), input.height.into());
    let _ = ironrdp_graphics::rle::decompress_8_bpp(input.src, &mut out, input.width.into(), input.height.into());
}

pub fn rdp6_encode_bitmap_stream(input: &BitmapInput<'_>) {
    use ironrdp_graphics::rdp6::{BitmapStreamEncoder, RgbAChannels, RgbChannels};

    let mut out = vec![0; input.src.len() * 2];

    let _ = BitmapStreamEncoder::new(input.width.into(), input.height.into()).encode_bitmap::<RgbChannels>(
        input.src,
        out.as_mut_slice(),
        false,
    );

    let _ = BitmapStreamEncoder::new(input.width.into(), input.height.into()).encode_bitmap::<RgbAChannels>(
        input.src,
        out.as_mut_slice(),
        true,
    );
}

pub fn rdp6_decode_bitmap_stream_to_rgb24(input: &BitmapInput<'_>) {
    use ironrdp_graphics::rdp6::BitmapStreamDecoder;

    let mut out = Vec::new();

    let _ = BitmapStreamDecoder::default().decode_bitmap_stream_to_rgb24(
        input.src,
        &mut out,
        usize::from(input.width),
        usize::from(input.height),
    );
}

pub fn cliprdr_format(input: &[u8]) {
    use ironrdp_cliprdr_format::bitmap::{dib_to_png, dibv5_to_png, png_to_cf_dib, png_to_cf_dibv5};
    use ironrdp_cliprdr_format::html::{cf_html_to_plain_html, plain_html_to_cf_html};

    let _ = png_to_cf_dib(input);
    let _ = png_to_cf_dibv5(input);

    let _ = dib_to_png(input);
    let _ = dibv5_to_png(input);

    let _ = cf_html_to_plain_html(input);

    if let Ok(input) = core::str::from_utf8(input) {
        let _ = plain_html_to_cf_html(input);
    }
}

pub fn channel_process(input: &[u8]) {
    use ironrdp_svc::SvcProcessor as _;

    let mut rdpdr = ironrdp_rdpdr::Rdpdr::new(Box::new(ironrdp_rdpdr::NoopRdpdrBackend), "Backend".to_owned())
        .with_smartcard(1)
        .with_drives(None);

    let _ = rdpdr.process(input);
}

pub fn cliprdr_channel_process(input: &[u8]) {
    use ironrdp_svc::SvcProcessor as _;

    let mut cliprdr = ironrdp_cliprdr::Cliprdr::<ironrdp_cliprdr::Client>::new(Box::new(NoopCliprdrFuzzBackend));
    let _ = cliprdr.process(input);
}

/// Minimal backend for fuzzing that enables file transfer capabilities
/// so the fuzzer can exercise lock, file list, and file contents paths.
#[derive(Debug)]
struct NoopCliprdrFuzzBackend;

ironrdp_core::impl_as_any!(NoopCliprdrFuzzBackend);

impl ironrdp_cliprdr::backend::CliprdrBackend for NoopCliprdrFuzzBackend {
    fn temporary_directory(&self) -> &str {
        "/tmp"
    }

    fn client_capabilities(&self) -> ironrdp_cliprdr::pdu::ClipboardGeneralCapabilityFlags {
        use ironrdp_cliprdr::pdu::ClipboardGeneralCapabilityFlags;
        ClipboardGeneralCapabilityFlags::STREAM_FILECLIP_ENABLED
            | ClipboardGeneralCapabilityFlags::CAN_LOCK_CLIPDATA
            | ClipboardGeneralCapabilityFlags::FILECLIP_NO_FILE_PATHS
            | ClipboardGeneralCapabilityFlags::HUGE_FILE_SUPPORT_ENABLED
    }

    fn on_ready(&mut self) {}
    fn on_request_format_list(&mut self) {}
    fn on_process_negotiated_capabilities(&mut self, _: ironrdp_cliprdr::pdu::ClipboardGeneralCapabilityFlags) {}
    fn on_remote_copy(&mut self, _: &[ironrdp_cliprdr::pdu::ClipboardFormat]) {}
    fn on_format_data_request(&mut self, _: ironrdp_cliprdr::pdu::FormatDataRequest) {}
    fn on_format_data_response(&mut self, _: ironrdp_cliprdr::pdu::FormatDataResponse<'_>) {}
    fn on_file_contents_request(&mut self, _: ironrdp_cliprdr::pdu::FileContentsRequest) {}
    fn on_file_contents_response(&mut self, _: ironrdp_cliprdr::pdu::FileContentsResponse<'_>) {}
    fn on_lock(&mut self, _: ironrdp_cliprdr::pdu::LockDataId) {}
    fn on_unlock(&mut self, _: ironrdp_cliprdr::pdu::LockDataId) {}

    // Fixed clock so fuzz runs are reproducible regardless of wall-clock timing
    fn now_ms(&self) -> u64 {
        0
    }
}

/// Wire-decode fuzz oracle for the ClearCodec codec.
///
/// Feeds arbitrary bytes through `ClearCodecDecoder::decode` with width/height
/// taken from the first four bytes (little-endian u16 each). Property: no
/// internal panic, no `unreachable!()`, no out-of-bounds access. Allocation
/// is bounded by the decoder's own `MAX_DECODE_PIXELS = 8192 * 8192` cap.
///
/// Decoder Err is silently dropped. The libFuzzer crash channel is reserved
/// for internal panics, asserts, and `unreachable!()` reached on
/// attacker-controlled inputs that the decoder accepted as well-formed.
#[expect(clippy::panic, reason = "panic is the libFuzzer bug-reporting mechanism")]
pub fn clearcodec_decode(data: &[u8]) {
    use ironrdp_graphics::clearcodec::ClearCodecDecoder;

    if data.len() < 4 {
        return;
    }
    let width = u16::from_le_bytes([data[0], data[1]]);
    let height = u16::from_le_bytes([data[2], data[3]]);
    let stream = &data[4..];

    let mut decoder = ClearCodecDecoder::new();
    let _ = decoder.decode(stream, width, height);
}

/// Round-trip fuzz oracle for ClearCodec: `encode -> decode`, check that
/// the round-tripped pixels match the input on the BGR channels.
///
/// **Why BGR-only and not BGRA:** ClearCodec is documented as lossless per
/// MS-RDPEGFX 2.2.4.1. The wire format is BGR (3 bytes/pixel); the encoder
/// accepts BGRA input and discards alpha, and the decoder fills decoded
/// alpha with 0xFF unconditionally. A BGRA-byte-exact round-trip is
/// therefore impossible by design for inputs whose alpha differs from 0xFF.
/// The first version of this oracle checked full BGRA equality and
/// immediately surfaced that on input `[1,0,1,0,4,255,4,4,255,4]`:
/// input BGRA=(4,255,4,4), decoded BGRA=(4,255,4,255). Real but
/// spec-consistent finding; should be documented on the codec's public
/// API (filed as a comment on PR #1174).
///
/// Input layout: `[width_le: u16][height_le: u16][bgr: u8 * width * height
/// * 3]`. Width and height are bounded so the oracle stays fast-fuzz-
/// friendly (pixel_count capped at 65,536 = 256x256; the decoder's
/// 8192*8192 cap is a separate, looser ceiling).
///
/// Three properties checked:
/// 1. Encoder never panics (it returns `Vec<u8>`, no `Result`, so any panic
///    is a real bug).
/// 2. Re-decoding the encoder's output succeeds (asymmetric impl gap if
///    not — separately reported).
/// 3. Re-decoded BGR triples exactly match the input BGR triples (alpha
///    bytes ignored, since the codec contract is BGR-lossless).
#[expect(clippy::panic, reason = "panic is the libFuzzer bug-reporting mechanism")]
pub fn clearcodec_round_trip(data: &[u8]) {
    use ironrdp_graphics::clearcodec::{ClearCodecDecoder, ClearCodecEncoder};

    if data.len() < 4 {
        return;
    }
    let width = u16::from_le_bytes([data[0], data[1]]);
    let height = u16::from_le_bytes([data[2], data[3]]);

    let w = usize::from(width);
    let h = usize::from(height);
    let Some(pixel_count) = w.checked_mul(h) else {
        return;
    };
    if pixel_count == 0 || pixel_count > 65_536 {
        return;
    }
    let bgr_size = pixel_count.saturating_mul(3);
    if data.len() < 4 + bgr_size {
        return;
    }
    let bgr_in = &data[4..4 + bgr_size];

    // Synthesize BGRA input with alpha = 0xFF for each pixel (the codec's
    // expected input shape; alpha is dropped by the encoder anyway).
    let mut bgra_in = Vec::with_capacity(pixel_count * 4);
    for px in 0..pixel_count {
        bgra_in.push(bgr_in[px * 3]);
        bgra_in.push(bgr_in[px * 3 + 1]);
        bgra_in.push(bgr_in[px * 3 + 2]);
        bgra_in.push(0xFF);
    }

    let mut encoder = ClearCodecEncoder::new();
    let encoded = encoder.encode(&bgra_in, width, height);

    let mut decoder = ClearCodecDecoder::new();
    let Ok(decoded) = decoder.decode(&encoded, width, height) else {
        panic!(
            "clearcodec_round_trip: decoder rejected encoder output (width={width}, height={height}, encoded_len={})",
            encoded.len(),
        );
    };

    if decoded.len() != bgra_in.len() {
        panic!(
            "clearcodec_round_trip: length mismatch (width={width}, height={height}, expected={}, decoded={})",
            bgra_in.len(),
            decoded.len(),
        );
    }

    // BGR-only equality check (skip alpha at every 4th byte).
    for px in 0..pixel_count {
        let off = px * 4;
        if decoded[off] != bgra_in[off]
            || decoded[off + 1] != bgra_in[off + 1]
            || decoded[off + 2] != bgra_in[off + 2]
        {
            panic!(
                "clearcodec_round_trip: BGR mismatch at pixel {px} \
                 (width={width}, height={height}): input=({:#x},{:#x},{:#x}) decoded=({:#x},{:#x},{:#x})",
                bgra_in[off],
                bgra_in[off + 1],
                bgra_in[off + 2],
                decoded[off],
                decoded[off + 1],
                decoded[off + 2],
            );
        }
    }
}
