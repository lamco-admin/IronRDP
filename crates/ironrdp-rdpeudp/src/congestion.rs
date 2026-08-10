//! NewReno congestion control for RDPEUDP2.
//!
//! MS-RDPEUDP Section 3.1.1 and MS-RDPEUDP2 Section 3.1.1.2.
//!
//! Implements a standard NewReno-style congestion controller with
//! slow start and congestion avoidance phases. Integrates with the
//! CN/CWR flag mechanism defined by MS-RDPEUDP:
//!
//! 1. Receiver detects loss → sets CN flag on next ACK.
//! 2. Sender receives CN → halves cwnd, sends CWR on next data.
//! 3. Receiver sees CWR → stops setting CN.
//! 4. Sender ignores CN flags whose snSourceAck < CWR sequence number.
//!
//! This ensures at most one congestion reaction per RTT.

/// Initial congestion window size in bytes.
///
/// MS-RDPEUDP2 doesn't specify an initial window. Quinn uses 14720
/// bytes (10 * 1200 + spare). We use 10 * 1232 = 12320, aligned to
/// the RDPEUDP2 MTU of 1232 bytes.
const INITIAL_WINDOW: u64 = 10 * 1232;

/// Minimum congestion window: 2 × MTU.
///
/// The window never drops below this, even after loss events.
const MIN_WINDOW: u64 = 2 * 1232;

/// NewReno congestion controller.
///
/// Manages the congestion window (cwnd), slow start threshold (ssthresh),
/// and loss epoch tracking via the CWR sequence number.
#[derive(Debug, Clone)]
pub(crate) struct CongestionControl {
    /// Congestion window in bytes.
    window: u64,

    /// Slow start threshold in bytes.
    /// Set to u64::MAX initially (all traffic starts in slow start).
    ssthresh: u64,

    /// Bytes acknowledged since the last window increase.
    /// Used for congestion avoidance: increase by 1 MTU per cwnd bytes ACKed.
    bytes_acked: u64,

    /// DataSeqNum of the packet that carried the most recent CWR flag.
    /// Used to limit congestion reaction to once per loss epoch.
    /// CN flags with snSourceAck < cwr_seq are stale and ignored.
    cwr_seq: Option<u64>,
}

impl CongestionControl {
    /// Create a new controller with the default initial window.
    pub(crate) fn new() -> Self {
        Self::with_initial_window(INITIAL_WINDOW)
    }

    /// Create a controller with a specified initial window size.
    pub(crate) fn with_initial_window(initial_window: u64) -> Self {
        Self {
            window: initial_window,
            ssthresh: u64::MAX,
            bytes_acked: 0,
            cwr_seq: None,
        }
    }

    /// Current congestion window in bytes.
    ///
    /// The caller should ensure `bytes_in_flight <= window()` before
    /// transmitting new data.
    pub(crate) fn window(&self) -> u64 {
        self.window
    }

    /// Whether the controller is in slow start phase.
    pub(crate) fn in_slow_start(&self) -> bool {
        self.window < self.ssthresh
    }

    /// Current slow start threshold.
    #[cfg(test)]
    pub(crate) fn ssthresh(&self) -> u64 {
        self.ssthresh
    }

    /// The CWR sequence number, if a congestion response is active.
    #[cfg(test)]
    pub(crate) fn cwr_seq(&self) -> Option<u64> {
        self.cwr_seq
    }

    /// Called when an ACK acknowledges new data.
    ///
    /// In slow start: window increases by `newly_acked_bytes` (doubles
    /// roughly every RTT).
    /// In congestion avoidance: window increases by approximately
    /// 1 MTU per cwnd bytes ACKed (linear increase).
    pub(crate) fn on_ack(&mut self, newly_acked_bytes: u64) {
        if self.in_slow_start() {
            // Slow start: exponential growth
            self.window += newly_acked_bytes;

            // Don't overshoot ssthresh
            if self.window >= self.ssthresh {
                self.bytes_acked = 0;
            }
        } else {
            // Congestion avoidance: linear growth
            // Increase by 1 MTU per cwnd bytes acknowledged
            self.bytes_acked += newly_acked_bytes;

            if self.bytes_acked >= self.window {
                self.bytes_acked -= self.window;
                self.window += 1232; // 1 MTU increase
            }
        }
    }

    /// Called when a loss event is detected.
    ///
    /// Halves the congestion window and updates ssthresh.
    /// Returns `true` if the window was actually reduced (false if
    /// we already reacted to this loss epoch).
    ///
    /// `loss_seq` is the DataSeqNum of the lost packet. If it's below
    /// the CWR sequence, this loss is from a previous epoch and we
    /// don't react again.
    pub(crate) fn on_loss(&mut self, loss_seq: u64) -> bool {
        // Check if this loss is from the current epoch
        if let Some(cwr) = self.cwr_seq {
            if loss_seq < cwr {
                // Loss from before the most recent CWR: already reacted
                return false;
            }
        }

        self.halve_window();
        true
    }

    /// Called when a CN (congestion notification) flag is received
    /// from the remote endpoint.
    ///
    /// `ack_seq` is the snSourceAck field from the packet carrying CN.
    /// Returns `true` if the window was reduced (i.e., this is a new
    /// congestion epoch), or `false` if the CN was stale.
    pub(crate) fn on_congestion_notification(&mut self, ack_seq: u64) -> bool {
        // Ignore stale CN: ack_seq < cwr_seq means this CN is about
        // a loss that predates our most recent CWR response.
        if let Some(cwr) = self.cwr_seq {
            if ack_seq < cwr {
                return false;
            }
        }

        self.halve_window();
        true
    }

    /// Record that a CWR flag was sent on the packet with this DataSeqNum.
    ///
    /// The connection state machine calls this after sending a packet
    /// with the CWR flag set. Future CN/loss events below this seq
    /// are treated as stale.
    pub(crate) fn set_cwr_seq(&mut self, seq: u64) {
        self.cwr_seq = Some(seq);
    }

    /// Halve the window (shared between on_loss and on_congestion_notification).
    fn halve_window(&mut self) {
        self.ssthresh = (self.window / 2).max(MIN_WINDOW);
        self.window = self.ssthresh;
        self.bytes_acked = 0;
    }
}

impl Default for CongestionControl {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state() {
        let cc = CongestionControl::new();
        assert_eq!(cc.window(), INITIAL_WINDOW);
        assert!(cc.in_slow_start()); // ssthresh = MAX
        assert_eq!(cc.cwr_seq(), None);
    }

    #[test]
    fn slow_start_doubles_per_rtt() {
        let mut cc = CongestionControl::new();
        let initial = cc.window();

        // ACK the entire window → window should roughly double
        cc.on_ack(initial);

        assert_eq!(cc.window(), initial * 2);
        assert!(cc.in_slow_start());
    }

    #[test]
    fn slow_start_growth() {
        let mut cc = CongestionControl::new();

        cc.on_ack(1232); // 1 MTU acked
        assert_eq!(cc.window(), INITIAL_WINDOW + 1232);
    }

    #[test]
    fn transition_to_congestion_avoidance_on_loss() {
        let mut cc = CongestionControl::new();
        let initial = cc.window();

        // Loss event
        cc.on_loss(1);
        assert!(!cc.in_slow_start()); // ssthresh = window/2 = initial/2, window = ssthresh
        assert_eq!(cc.window(), initial / 2);
        assert_eq!(cc.ssthresh(), initial / 2);
    }

    #[test]
    fn congestion_avoidance_linear_growth() {
        let mut cc = CongestionControl::with_initial_window(12320);

        // Force into congestion avoidance
        cc.on_loss(1);
        let window_after_loss = cc.window(); // 6160

        // In congestion avoidance, need to ACK ~cwnd bytes to gain 1 MTU
        cc.on_ack(window_after_loss);

        // Window should increase by 1 MTU
        assert_eq!(cc.window(), window_after_loss + 1232);
    }

    #[test]
    fn window_floor_on_loss() {
        let mut cc = CongestionControl::with_initial_window(MIN_WINDOW);

        cc.on_loss(1);
        // Window should not drop below MIN_WINDOW
        assert_eq!(cc.window(), MIN_WINDOW);
    }

    #[test]
    fn on_loss_once_per_epoch() {
        let mut cc = CongestionControl::new();
        let initial = cc.window();

        // First loss
        assert!(cc.on_loss(5));
        let after_first = cc.window();
        assert_eq!(after_first, initial / 2);

        // Set CWR at seq 10
        cc.set_cwr_seq(10);

        // Second loss at seq 7 (before CWR) → stale, no reaction
        assert!(!cc.on_loss(7));
        assert_eq!(cc.window(), after_first);

        // Third loss at seq 12 (after CWR) → new epoch, react
        assert!(cc.on_loss(12));
        assert_eq!(cc.window(), after_first / 2);
    }

    #[test]
    fn congestion_notification_reduces_window() {
        let mut cc = CongestionControl::new();
        let initial = cc.window();

        assert!(cc.on_congestion_notification(1));
        assert_eq!(cc.window(), initial / 2);
    }

    #[test]
    fn stale_cn_ignored() {
        let mut cc = CongestionControl::new();

        cc.on_congestion_notification(5);
        cc.set_cwr_seq(10);

        let window_before = cc.window();

        // CN with ack_seq=7 < cwr_seq=10 → stale
        assert!(!cc.on_congestion_notification(7));
        assert_eq!(cc.window(), window_before);
    }

    #[test]
    fn cn_after_cwr_epoch_reacts() {
        let mut cc = CongestionControl::new();

        cc.on_congestion_notification(5);
        cc.set_cwr_seq(10);

        let window_before = cc.window();

        // CN with ack_seq=15 >= cwr_seq=10 → new epoch
        assert!(cc.on_congestion_notification(15));
        assert!(cc.window() < window_before);
    }

    #[test]
    fn repeated_loss_converges_to_floor() {
        let mut cc = CongestionControl::new();

        for seq in 0..50 {
            cc.on_loss(seq);
            cc.set_cwr_seq(seq + 1);
        }

        assert_eq!(cc.window(), MIN_WINDOW);
    }

    #[test]
    fn default_matches_new() {
        let from_new = CongestionControl::new();
        let from_default = CongestionControl::default();
        assert_eq!(from_new.window(), from_default.window());
        assert_eq!(from_new.ssthresh(), from_default.ssthresh());
    }

    #[test]
    fn custom_initial_window() {
        let cc = CongestionControl::with_initial_window(5000);
        assert_eq!(cc.window(), 5000);
    }

    #[test]
    fn slow_start_to_avoidance_boundary() {
        let mut cc = CongestionControl::new();
        let initial = cc.window();

        // Trigger loss to set ssthresh
        cc.on_loss(1);
        cc.set_cwr_seq(2);
        let ssthresh = cc.ssthresh();
        assert_eq!(ssthresh, initial / 2);

        // Window is at ssthresh, so we're in congestion avoidance
        assert!(!cc.in_slow_start());

        // If window somehow goes below ssthresh, we'd be in slow start
        // (this doesn't happen normally, but tests the boundary)
    }
}
