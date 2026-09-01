#pragma once

#include "driftless/road_graph.h"

#include <cstddef>
#include <vector>

namespace driftless {

// Newson & Krumm (2009), "Hidden Markov Map Matching Through Noise and
// Sparseness", in local NED rather than lat/lon.
//
// The model: hidden states are candidate road segments near each observation.
// The emission term says a position observation should be close to the road you
// are actually on. The transition term says that between two consecutive
// observations, the distance you travelled ALONG THE ROADS should be about the
// same as the straight-line distance between the observations -- if they differ
// wildly, that pairing implies teleporting or an absurd detour, and Viterbi
// will prefer a different pairing.
//
// That transition term is what makes this worth doing instead of just snapping
// to the nearest road: nearest-road picks the wrong carriageway of a dual
// carriageway roughly half the time, and flips between them at random. The HMM
// picks the sequence that is collectively plausible.
struct MapMatchParams {
    double candidate_radius_m = 50.0;
    std::size_t max_candidates_per_observation = 8;

    // Newson & Krumm's sigma_z: how far a position estimate is expected to sit
    // from the true road centreline. Not GNSS accuracy alone -- it also absorbs
    // lane offset and centreline digitisation error in the map itself.
    double emission_sigma_m = 10.0;

    // Newson & Krumm's beta: tolerance on the mismatch between straight-line
    // and on-road distance. Small beta punishes detours hard.
    double transition_beta_m = 5.0;

    double max_route_distance_m = 2000.0;
};

struct MapMatchResult {
    bool matched = false;
    int segment_id = -1;
    int way_id = -1;
    Vec3 snapped_position = Vec3::Zero();
    // Unit vector along the matched road, horizontal. The filter needs this to
    // know which direction the map actually constrains.
    Vec3 segment_direction = Vec3::UnitX();
    double distance_to_road_m = 0.0;
};

class HmmMapMatcher {
public:
    HmmMapMatcher(const RoadGraph& graph, const MapMatchParams& params);

    // Offline Viterbi with backtracking over a whole trajectory. More accurate
    // than step(), because a later observation can correct an earlier ambiguous
    // match. Use for evaluation and for the position plots in the proposal.
    [[nodiscard]] std::vector<MapMatchResult> match(
        const std::vector<Vec3>& positions) const;

    // Causal forward decoding, one observation at a time. This is what the
    // filter can actually use online -- it cannot wait for the future -- and it
    // is strictly weaker than match(): an ambiguity resolved three observations
    // later is resolved too late to help the estimate that was already emitted.
    MapMatchResult step(const Vec3& position);
    void reset();

private:
    struct Hypothesis {
        SegmentProjection projection;
        double log_probability = 0.0;
    };

    [[nodiscard]] double emissionLogProbability(double distance_m) const;
    [[nodiscard]] double transitionLogProbability(
        const SegmentProjection& from, const SegmentProjection& to,
        double straight_line_m) const;
    [[nodiscard]] MapMatchResult toResult(const SegmentProjection& projection) const;

    const RoadGraph* graph_;
    MapMatchParams params_;

    std::vector<Hypothesis> hypotheses_;
    Vec3 last_position_ = Vec3::Zero();
    bool has_last_position_ = false;
};

}  // namespace driftless
