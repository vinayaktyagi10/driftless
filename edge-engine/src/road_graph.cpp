#include "driftless/road_graph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace driftless {
namespace {
constexpr double kDegToRad = M_PI / 180.0;
}  // namespace

LocalTangentFrame::LocalTangentFrame(GeoPoint origin) : origin_(origin) {
    // Standard WGS84 series for the length of a degree at a given latitude.
    // Cheaper and more transparent than a full geodesic library, and accurate
    // to well under a metre per degree at the scales used here.
    const double phi = origin.latitude_deg * kDegToRad;
    metres_per_degree_latitude_ =
        111132.92 - 559.82 * std::cos(2 * phi) + 1.175 * std::cos(4 * phi) -
        0.0023 * std::cos(6 * phi);
    metres_per_degree_longitude_ = 111412.84 * std::cos(phi) -
                                   93.5 * std::cos(3 * phi) +
                                   0.118 * std::cos(5 * phi);
}

Vec3 LocalTangentFrame::toNed(GeoPoint point) const {
    return Vec3((point.latitude_deg - origin_.latitude_deg) *
                    metres_per_degree_latitude_,
                (point.longitude_deg - origin_.longitude_deg) *
                    metres_per_degree_longitude_,
                0.0);
}

GeoPoint LocalTangentFrame::toGeo(const Vec3& ned) const {
    return GeoPoint{origin_.latitude_deg + ned.x() / metres_per_degree_latitude_,
                    origin_.longitude_deg + ned.y() / metres_per_degree_longitude_};
}

Vec3 RoadSegment::direction() const {
    const Vec3 delta = end - start;
    const double norm = delta.norm();
    if (norm <= 0.0) return Vec3::UnitX();
    return delta / norm;
}

int RoadGraph::addNode(const Vec3& ned_position) {
    nodes_.push_back(ned_position);
    adjacency_valid_ = false;
    return static_cast<int>(nodes_.size()) - 1;
}

int RoadGraph::addSegment(int start_node, int end_node, int way_id) {
    if (start_node < 0 || end_node < 0 ||
        start_node >= static_cast<int>(nodes_.size()) ||
        end_node >= static_cast<int>(nodes_.size())) {
        return -1;
    }
    RoadSegment segment;
    segment.id = static_cast<int>(segments_.size());
    segment.way_id = way_id;
    segment.start_node = start_node;
    segment.end_node = end_node;
    segment.start = nodes_[start_node];
    segment.end = nodes_[end_node];
    // Duplicate consecutive nodes are common in raw OSM ways and would give a
    // NaN direction, so they are dropped rather than stored.
    if (segment.length() <= 0.0) return -1;

    segments_.push_back(segment);
    adjacency_valid_ = false;
    return segment.id;
}

SegmentProjection RoadGraph::project(const RoadSegment& segment,
                                     const Vec3& position) {
    const Vec3 delta = segment.end - segment.start;
    const double length_squared = delta.squaredNorm();

    double t = 0.0;
    if (length_squared > 0.0) {
        t = (position - segment.start).dot(delta) / length_squared;
        // Clamped, so a point beyond either end snaps to that endpoint rather
        // than to an imaginary extension of the road.
        t = std::clamp(t, 0.0, 1.0);
    }

    SegmentProjection projection;
    projection.segment_id = segment.id;
    projection.point = segment.start + t * delta;
    projection.along_m = t * segment.length();
    // Horizontal distance only: a map has no opinion about altitude, and
    // including a vertical term would let a bad height estimate reject the
    // correct road.
    const Vec3 offset = position - projection.point;
    projection.distance_m = std::hypot(offset.x(), offset.y());
    return projection;
}

std::vector<SegmentProjection> RoadGraph::candidatesNear(
    const Vec3& position, double radius_m, std::size_t max_candidates) const {
    std::vector<SegmentProjection> candidates;
    for (const auto& segment : segments_) {
        SegmentProjection projection = project(segment, position);
        if (projection.distance_m <= radius_m) candidates.push_back(projection);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const SegmentProjection& a, const SegmentProjection& b) {
                  return a.distance_m < b.distance_m;
              });
    if (candidates.size() > max_candidates) candidates.resize(max_candidates);
    return candidates;
}

void RoadGraph::ensureAdjacency() const {
    if (adjacency_valid_) return;
    adjacency_.assign(nodes_.size(), {});
    for (const auto& segment : segments_) {
        const double length = segment.length();
        adjacency_[segment.start_node].push_back({segment.end_node, length});
        adjacency_[segment.end_node].push_back({segment.start_node, length});
    }
    adjacency_valid_ = true;
}

double RoadGraph::routeDistance(const SegmentProjection& from,
                                const SegmentProjection& to,
                                double max_distance_m) const {
    if (from.segment_id < 0 || to.segment_id < 0) return -1.0;
    const RoadSegment& from_segment = segments_[from.segment_id];
    const RoadSegment& to_segment = segments_[to.segment_id];

    // Same segment: the road distance is just the difference in arc length.
    // Handled separately because the graph search below always leaves via a
    // node and would otherwise route the long way round the block.
    if (from.segment_id == to.segment_id) {
        return std::abs(to.along_m - from.along_m);
    }

    ensureAdjacency();

    // Dijkstra from the two ends of the source segment, seeded with the cost of
    // reaching each from the projected point.
    std::vector<double> distance(nodes_.size(),
                                 std::numeric_limits<double>::infinity());
    using Entry = std::pair<double, int>;  // (cost, node)
    std::priority_queue<Entry, std::vector<Entry>, std::greater<>> frontier;

    const auto seed = [&](int node, double cost) {
        if (cost < distance[node]) {
            distance[node] = cost;
            frontier.emplace(cost, node);
        }
    };
    seed(from_segment.start_node, from.along_m);
    seed(from_segment.end_node, from_segment.length() - from.along_m);

    const double to_start_cost = to.along_m;
    const double to_end_cost = to_segment.length() - to.along_m;

    double best = std::numeric_limits<double>::infinity();
    while (!frontier.empty()) {
        const auto [cost, node] = frontier.top();
        frontier.pop();
        if (cost > distance[node]) continue;
        // Everything still in the frontier costs at least this much, so once
        // the frontier passes the budget or the best answer, we are done.
        if (cost > max_distance_m || cost >= best) break;

        if (node == to_segment.start_node) best = std::min(best, cost + to_start_cost);
        if (node == to_segment.end_node) best = std::min(best, cost + to_end_cost);

        for (const auto& neighbour : adjacency_[node]) {
            const double next = cost + neighbour.length;
            if (next < distance[neighbour.node]) {
                distance[neighbour.node] = next;
                frontier.emplace(next, neighbour.node);
            }
        }
    }

    if (!std::isfinite(best) || best > max_distance_m) return -1.0;
    return best;
}

}  // namespace driftless
