#include "driftless/hmm_map_matcher.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace driftless {
namespace {
// Log-probabilities are only ever compared, never exponentiated, so the
// normalizing constants of both distributions cancel and are omitted.
constexpr double kImpossible = -std::numeric_limits<double>::infinity();
}  // namespace

HmmMapMatcher::HmmMapMatcher(const RoadGraph& graph, const MapMatchParams& params)
    : graph_(&graph), params_(params) {}

double HmmMapMatcher::emissionLogProbability(double distance_m) const {
    const double scaled = distance_m / params_.emission_sigma_m;
    return -0.5 * scaled * scaled;
}

double HmmMapMatcher::transitionLogProbability(const SegmentProjection& from,
                                               const SegmentProjection& to,
                                               double straight_line_m) const {
    const double route_m =
        graph_->routeDistance(from, to, params_.max_route_distance_m);
    // No route within budget: not a low-probability transition but an
    // impossible one. Keeping it as -inf rather than a large penalty stops
    // Viterbi from ever preferring a physically disconnected path.
    if (route_m < 0.0) return kImpossible;
    return -std::abs(straight_line_m - route_m) / params_.transition_beta_m;
}

MapMatchResult HmmMapMatcher::toResult(const SegmentProjection& projection) const {
    MapMatchResult result;
    if (projection.segment_id < 0) return result;
    const RoadSegment& segment = graph_->segments()[projection.segment_id];
    result.matched = true;
    result.segment_id = projection.segment_id;
    result.way_id = segment.way_id;
    result.snapped_position = projection.point;
    result.segment_direction = segment.direction();
    result.distance_to_road_m = projection.distance_m;
    return result;
}

void HmmMapMatcher::reset() {
    hypotheses_.clear();
    has_last_position_ = false;
}

MapMatchResult HmmMapMatcher::step(const Vec3& position) {
    const auto candidates = graph_->candidatesNear(
        position, params_.candidate_radius_m, params_.max_candidates_per_observation);
    if (candidates.empty()) {
        // Off-network: drop the hypothesis set rather than carrying stale
        // beliefs across a gap. Re-acquisition starts clean.
        reset();
        return MapMatchResult{};
    }

    std::vector<Hypothesis> next;
    next.reserve(candidates.size());

    if (hypotheses_.empty() || !has_last_position_) {
        for (const auto& candidate : candidates) {
            next.push_back({candidate, emissionLogProbability(candidate.distance_m)});
        }
    } else {
        const double straight_line = (position - last_position_).norm();
        for (const auto& candidate : candidates) {
            double best = kImpossible;
            for (const auto& previous : hypotheses_) {
                if (previous.log_probability == kImpossible) continue;
                const double transition = transitionLogProbability(
                    previous.projection, candidate, straight_line);
                if (transition == kImpossible) continue;
                best = std::max(best, previous.log_probability + transition);
            }
            if (best == kImpossible) continue;
            next.push_back({candidate, best + emissionLogProbability(candidate.distance_m)});
        }
        if (next.empty()) {
            // Every transition was impossible -- typically a jump across a
            // river or a long GNSS gap. Restart from the emission term alone
            // instead of reporting nothing.
            for (const auto& candidate : candidates) {
                next.push_back({candidate, emissionLogProbability(candidate.distance_m)});
            }
        }
    }

    // Renormalize so scores cannot drift toward -inf over a long trajectory.
    // Only differences matter, so subtracting the maximum changes nothing.
    const double best_score =
        std::max_element(next.begin(), next.end(),
                         [](const Hypothesis& a, const Hypothesis& b) {
                             return a.log_probability < b.log_probability;
                         })
            ->log_probability;
    for (auto& hypothesis : next) hypothesis.log_probability -= best_score;

    hypotheses_ = std::move(next);
    last_position_ = position;
    has_last_position_ = true;

    const auto best = std::max_element(
        hypotheses_.begin(), hypotheses_.end(),
        [](const Hypothesis& a, const Hypothesis& b) {
            return a.log_probability < b.log_probability;
        });
    return toResult(best->projection);
}

std::vector<MapMatchResult> HmmMapMatcher::match(
    const std::vector<Vec3>& positions) const {
    std::vector<MapMatchResult> results(positions.size());
    if (positions.empty()) return results;

    // Full Viterbi: per-observation candidate lists, scores, and backpointers.
    std::vector<std::vector<SegmentProjection>> candidates(positions.size());
    std::vector<std::vector<double>> scores(positions.size());
    std::vector<std::vector<int>> backpointers(positions.size());

    for (std::size_t t = 0; t < positions.size(); ++t) {
        candidates[t] = graph_->candidatesNear(positions[t], params_.candidate_radius_m,
                                               params_.max_candidates_per_observation);
        scores[t].assign(candidates[t].size(), kImpossible);
        backpointers[t].assign(candidates[t].size(), -1);
    }

    // Forward pass.
    std::size_t first = positions.size();
    for (std::size_t t = 0; t < positions.size(); ++t) {
        if (candidates[t].empty()) continue;
        if (first == positions.size()) {
            for (std::size_t i = 0; i < candidates[t].size(); ++i) {
                scores[t][i] = emissionLogProbability(candidates[t][i].distance_m);
            }
            first = t;
            continue;
        }
        // Previous observation that actually had candidates.
        std::size_t previous = t;
        while (previous > first && candidates[previous - 1].empty()) --previous;
        --previous;

        const double straight_line = (positions[t] - positions[previous]).norm();
        for (std::size_t i = 0; i < candidates[t].size(); ++i) {
            double best = kImpossible;
            int best_index = -1;
            for (std::size_t j = 0; j < candidates[previous].size(); ++j) {
                if (scores[previous][j] == kImpossible) continue;
                const double transition = transitionLogProbability(
                    candidates[previous][j], candidates[t][i], straight_line);
                if (transition == kImpossible) continue;
                const double total = scores[previous][j] + transition;
                if (total > best) {
                    best = total;
                    best_index = static_cast<int>(j);
                }
            }
            if (best == kImpossible) {
                // Unreachable from every predecessor: restart the chain here
                // rather than dropping the observation entirely.
                scores[t][i] = emissionLogProbability(candidates[t][i].distance_m);
                backpointers[t][i] = -1;
            } else {
                scores[t][i] = best + emissionLogProbability(candidates[t][i].distance_m);
                backpointers[t][i] = best_index;
            }
        }

        const double best_score = *std::max_element(scores[t].begin(), scores[t].end());
        if (std::isfinite(best_score)) {
            for (auto& score : scores[t]) {
                if (score != kImpossible) score -= best_score;
            }
        }
    }
    if (first == positions.size()) return results;

    // Backward pass from the last observation that had candidates.
    std::size_t last = positions.size() - 1;
    while (last > first && candidates[last].empty()) --last;

    int index = static_cast<int>(std::max_element(scores[last].begin(),
                                                  scores[last].end()) -
                                 scores[last].begin());
    for (std::size_t t = last + 1; t-- > first;) {
        if (candidates[t].empty() || index < 0) continue;
        results[t] = toResult(candidates[t][static_cast<std::size_t>(index)]);
        index = backpointers[t][static_cast<std::size_t>(index)];
        if (index < 0 && t > first) {
            // Chain restarted here; pick up the best candidate of the previous
            // observation on its own merits.
            std::size_t previous = t - 1;
            while (previous > first && candidates[previous].empty()) --previous;
            if (!candidates[previous].empty()) {
                index = static_cast<int>(std::max_element(scores[previous].begin(),
                                                          scores[previous].end()) -
                                         scores[previous].begin());
            }
        }
    }
    return results;
}

}  // namespace driftless
