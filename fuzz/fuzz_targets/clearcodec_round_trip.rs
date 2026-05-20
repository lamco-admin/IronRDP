#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    ironrdp_fuzzing::oracles::clearcodec_round_trip(data);
});
