#pragma once

#include "driftless/types.h"

#include <cstddef>
#include <vector>

namespace driftless {

struct GeoPoint {
    double latitude_deg = 0.0;
    double longitude_deg = 0.0;
};

// Equirectangular projection about a fixed origin, giving the same local NED
// frame the filter works in. Valid because a map-matching working set spans
// kilometres, not degrees: over that range the error from ignoring earth
// curvature is centimetres, far below the lane-level resolution that matters.
// It is NOT valid for country-scale extracts, which is why the origin is
// mandatory and explicit rather than defaulted.
class LocalTangentFrame {
public:
    explicit LocalTangentFrame(GeoPoint origin);

    [[nodiscard]] Vec3 toNed(GeoPoint point) const;
    [[nodiscard]] GeoPoint toGeo(const Vec3& ned) const;
    [[nodiscard]] GeoPoint origin() const { return origin_; }

private:
    GeoPoint origin_;
    double metres_per_degree_latitude_;
    double metres_per_degree_longitude_;
};

struct RoadSegment {
    int id = -1;
    int way_id = -1;
    int start_node = -1;
    int end_node = -1;
    Vec3 start = Vec3::Zero();
    Vec3 end = Vec3::Zero();

    [[nodiscard]] double length() const { return (end - start).norm(); }
    // Unit vector along the segment. Zero-length segments return +North rather
    // than a NaN; callers should filter those out at load time instead.
    [[nodiscard]] Vec3 direction() const;
};

// The closest point on one segment to a query position.
struct SegmentProjection {
    int segment_id = -1;
    Vec3 point = Vec3::Zero();     // the snapped position, in NED
    double distance_m = 0.0;       // perpendicular distance from the query
    double along_m = 0.0;          // arc length from the segment's start node
};

// A road network in local NED coordinates.
//
// Segments are stored undirected: both travel directions are added to the
// adjacency used for routing. Real one-way handling needs the OSM `oneway` tag
// and turn restrictions, which matters for route distance in dense city
// centres and is a known simplification here.
// TODO: honour oneway/turn restrictions once matching runs on a real extract.
class RoadGraph {
public:
    int addNode(const Vec3& ned_position);
    // Returns the new segment id, or -1 if the segment has zero length.
    int addSegment(int start_node, int end_node, int way_id);

    [[nodiscard]] const std::vector<Vec3>& nodes() const { return nodes_; }
    [[nodiscard]] const std::vector<RoadSegment>& segments() const { return segments_; }
    [[nodiscard]] std::size_t segmentCount() const { return segments_.size(); }

    // Project a point onto one segment, clamping to the segment's extent.
    [[nodiscard]] static SegmentProjection project(const RoadSegment& segment,
                                                   const Vec3& position);

    // Segments whose closest point lies within radius_m, nearest first, capped
    // at max_candidates.
    // TODO: linear scan over all segments. Fine for the working set of an
    // urban extract; needs a uniform grid or R-tree before it sees a whole
    // city at 10Hz.
    [[nodiscard]] std::vector<SegmentProjection> candidatesNear(
        const Vec3& position, double radius_m, std::size_t max_candidates) const;

    // Shortest distance along the road network between two projected points.
    // Returns a negative value if no route shorter than max_distance_m exists,
    // which Viterbi treats as an impossible transition rather than an error.
    [[nodiscard]] double routeDistance(const SegmentProjection& from,
                                       const SegmentProjection& to,
                                       double max_distance_m) const;

private:
    void ensureAdjacency() const;

    std::vector<Vec3> nodes_;
    std::vector<RoadSegment> segments_;

    struct AdjacentNode {
        int node = -1;
        double length = 0.0;
    };
    // Built lazily on first routing query and invalidated by any mutation.
    mutable std::vector<std::vector<AdjacentNode>> adjacency_;
    mutable bool adjacency_valid_ = false;
};

}  // namespace driftless
