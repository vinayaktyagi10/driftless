#pragma once

namespace driftless {

// Newson & Krumm HMM map matching against an offline OSM extract, mirroring
// android/.../mapmatch/HmmMapMatcher.kt. Kept separate from the Android
// implementation because this engine also has to serve non-mobile
// deployments where "offline map" may be a different extract entirely.
class HmmMapMatcher {
public:
    HmmMapMatcher() = default;
    // TODO: load OSM extract, viterbi over candidate road segments
};

}  // namespace driftless
