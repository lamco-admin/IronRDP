//! Monotonic time readings supplied by the caller.
//!
//! The connection state machine never reads a clock. It receives the current
//! instant on every call that can advance time, and reports when it next wants
//! to be woken through [`RdpeudpConnection::poll_timeout`].
//!
//! `std::time::Instant` is deliberately not used here. `Instant::now` panics
//! on `wasm32-unknown-unknown`, which this crate compiles for, and a state
//! machine that read a clock itself would measure how quickly it drained an
//! already-filled buffer rather than when the datagram actually arrived.
//!
//! [`RdpeudpConnection::poll_timeout`]: crate::RdpeudpConnection::poll_timeout

use core::ops::Add;
use core::time::Duration;

/// A monotonic instant, in milliseconds, from an epoch chosen by the caller.
///
/// The epoch is arbitrary and carries no meaning; only differences between two
/// instants do. Millisecond resolution is well below the shortest interval the
/// protocol asks us to measure, the delayed-ACK timer of [MS-RDPEUDP2] 3.1.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MonotonicInstant(u64);

impl MonotonicInstant {
    /// Builds an instant from a monotonic millisecond reading.
    #[must_use]
    pub fn from_millis(milliseconds: u64) -> Self {
        Self(milliseconds)
    }

    /// The reading this instant was built from.
    #[must_use]
    pub fn as_millis(self) -> u64 {
        self.0
    }

    /// The time elapsed since `earlier`, saturating at zero if the clock went
    /// backwards or the arguments were transposed.
    #[must_use]
    pub fn duration_since(self, earlier: Self) -> Duration {
        Duration::from_millis(self.0.saturating_sub(earlier.0))
    }

    /// This instant advanced by `duration`, saturating at the end of the
    /// representable range.
    ///
    /// Saturating is the right behavior for the only caller: a deadline that
    /// cannot be represented is one that never fires, which is what a timer set
    /// beyond the end of time should do.
    #[must_use]
    pub fn saturating_add(self, duration: Duration) -> Self {
        let milliseconds = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
        Self(self.0.saturating_add(milliseconds))
    }
}

impl Add<Duration> for MonotonicInstant {
    type Output = Self;

    fn add(self, rhs: Duration) -> Self {
        self.saturating_add(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_since_measures_forward_difference() {
        let earlier = MonotonicInstant::from_millis(1_000);
        let later = MonotonicInstant::from_millis(1_750);

        assert_eq!(later.duration_since(earlier), Duration::from_millis(750));
    }

    #[test]
    fn duration_since_saturates_when_transposed() {
        let earlier = MonotonicInstant::from_millis(1_000);
        let later = MonotonicInstant::from_millis(1_750);

        assert_eq!(earlier.duration_since(later), Duration::ZERO);
    }

    #[test]
    fn add_advances_by_the_duration() {
        let base = MonotonicInstant::from_millis(500);

        assert_eq!(base + Duration::from_millis(250), MonotonicInstant::from_millis(750));
    }

    #[test]
    fn add_saturates_at_the_end_of_the_range() {
        let base = MonotonicInstant::from_millis(u64::MAX - 1);

        assert_eq!(base + Duration::from_secs(60), MonotonicInstant::from_millis(u64::MAX));
    }

    #[test]
    fn add_saturates_when_the_duration_itself_overflows() {
        let base = MonotonicInstant::from_millis(0);

        assert_eq!(base + Duration::MAX, MonotonicInstant::from_millis(u64::MAX));
    }

    #[test]
    fn instants_order_by_reading() {
        assert!(MonotonicInstant::from_millis(1) < MonotonicInstant::from_millis(2));
    }
}
