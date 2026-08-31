#pragma once

namespace driftless {

// Same UKF design as android/.../fusion/UkfFusionEngine.kt, retuned for
// ~200Hz FOG IMU input instead of ~100Hz consumer MEMS. State vector and
// filter math should stay in sync across the two implementations even
// though the code doesn't share a runtime.
class UkfFusionEngine {
public:
    UkfFusionEngine() = default;
    // TODO: predict(ImuSample), updateGnss(GnssFix), updateMapMatch(Position)
};

}  // namespace driftless
