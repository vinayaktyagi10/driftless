// Road graph geometry, routing, and the Newson & Krumm HMM.

#include "driftless/hmm_map_matcher.h"
#include "driftless/road_graph.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <random>

namespace {

using driftless::GeoPoint;
using driftless::HmmMapMatcher;
using driftless::LocalTangentFrame;
using driftless::MapMatchParams;
using driftless::RoadGraph;
using driftless::RoadSegment;
using driftless::Vec3;

// A straight road along North at a given East offset, split into 100 m
// segments. Returns its way id.
int addStraightRoad(RoadGraph& graph, double east_m, double length_m, int way_id) {
    int previous = graph.addNode(Vec3(0.0, east_m, 0.0));
    for (double north = 100.0; north <= length_m + 1e-9; north += 100.0) {
        const int next = graph.addNode(Vec3(north, east_m, 0.0));
        graph.addSegment(previous, next, way_id);
        previous = next;
    }
    return way_id;
}

// --- Projection ------------------------------------------------------------

TEST(LocalTangentFrame, RoundTripsAndHasPlausibleScale) {
    const LocalTangentFrame frame(GeoPoint{26.8467, 75.8056});  // Jaipur
    const GeoPoint point{26.8567, 75.8156};
    const Vec3 ned = frame.toNed(point);
    const GeoPoint back = frame.toGeo(ned);

    EXPECT_NEAR(back.latitude_deg, point.latitude_deg, 1e-12);
    EXPECT_NEAR(back.longitude_deg, point.longitude_deg, 1e-12);

    // 0.01 degree of latitude is ~1.11 km anywhere on earth.
    EXPECT_NEAR(ned.x(), 1110.0, 5.0);
    // A degree of longitude shortens with cos(latitude): ~0.89 of a degree of
    // latitude at 26.8 degrees North.
    EXPECT_NEAR(ned.y() / ned.x(), std::cos(26.8467 * M_PI / 180.0), 0.01);
    EXPECT_EQ(ned.z(), 0.0);
}

TEST(RoadGraph, ProjectsPerpendicularlyAndClampsBeyondTheEnds) {
    RoadGraph graph;
    const int a = graph.addNode(Vec3(0.0, 0.0, 0.0));
    const int b = graph.addNode(Vec3(100.0, 0.0, 0.0));
    const int id = graph.addSegment(a, b, 1);
    ASSERT_GE(id, 0);
    const RoadSegment& segment = graph.segments()[id];

    const auto middle = RoadGraph::project(segment, Vec3(50.0, 7.0, 0.0));
    EXPECT_NEAR(middle.point.x(), 50.0, 1e-12);
    EXPECT_NEAR(middle.distance_m, 7.0, 1e-12);
    EXPECT_NEAR(middle.along_m, 50.0, 1e-12);

    // Past the far end: snaps to the endpoint, not to an extension of the road.
    const auto beyond = RoadGraph::project(segment, Vec3(180.0, 0.0, 0.0));
    EXPECT_NEAR(beyond.point.x(), 100.0, 1e-12);
    EXPECT_NEAR(beyond.distance_m, 80.0, 1e-12);
    EXPECT_NEAR(beyond.along_m, 100.0, 1e-12);
}

TEST(RoadGraph, RejectsZeroLengthSegments) {
    // Duplicate consecutive nodes are common in raw OSM and would give a NaN
    // direction if stored.
    RoadGraph graph;
    const int a = graph.addNode(Vec3(5.0, 5.0, 0.0));
    const int b = graph.addNode(Vec3(5.0, 5.0, 0.0));
    EXPECT_EQ(graph.addSegment(a, b, 1), -1);
    EXPECT_EQ(graph.segmentCount(), 0u);
}

TEST(RoadGraph, CandidateSearchRespectsRadiusOrderingAndCap) {
    RoadGraph graph;
    addStraightRoad(graph, 0.0, 300.0, 1);
    addStraightRoad(graph, 20.0, 300.0, 2);
    addStraightRoad(graph, 500.0, 300.0, 3);  // far away

    const auto candidates = graph.candidatesNear(Vec3(150.0, 2.0, 0.0), 50.0, 8);
    ASSERT_GE(candidates.size(), 2u);
    EXPECT_LT(candidates[0].distance_m, candidates[1].distance_m);
    EXPECT_NEAR(candidates[0].distance_m, 2.0, 1e-9);
    for (const auto& candidate : candidates) EXPECT_LE(candidate.distance_m, 50.0);

    EXPECT_LE(graph.candidatesNear(Vec3(150.0, 2.0, 0.0), 1000.0, 3).size(), 3u);
}

// --- Routing ---------------------------------------------------------------

TEST(RoadGraph, RouteDistanceAlongASingleRoadIsArcLength) {
    RoadGraph graph;
    addStraightRoad(graph, 0.0, 500.0, 1);

    const auto from = graph.candidatesNear(Vec3(50.0, 0.0, 0.0), 10.0, 1).front();
    const auto to = graph.candidatesNear(Vec3(430.0, 0.0, 0.0), 10.0, 1).front();
    EXPECT_NEAR(graph.routeDistance(from, to, 2000.0), 380.0, 1e-6);
}

TEST(RoadGraph, RouteDistanceGoesAroundCornersNotThroughThem) {
    // An L-shaped road: straight-line distance is the hypotenuse, but the road
    // distance is the two legs. This difference is exactly what the HMM's
    // transition term is built to notice.
    RoadGraph graph;
    const int corner = graph.addNode(Vec3(0.0, 0.0, 0.0));
    const int north = graph.addNode(Vec3(100.0, 0.0, 0.0));
    const int east = graph.addNode(Vec3(0.0, 100.0, 0.0));
    graph.addSegment(corner, north, 1);
    graph.addSegment(corner, east, 2);

    const auto from = graph.candidatesNear(Vec3(100.0, 0.0, 0.0), 5.0, 1).front();
    const auto to = graph.candidatesNear(Vec3(0.0, 100.0, 0.0), 5.0, 1).front();

    EXPECT_NEAR(graph.routeDistance(from, to, 2000.0), 200.0, 1e-6);
    EXPECT_NEAR((Vec3(100.0, 0.0, 0.0) - Vec3(0.0, 100.0, 0.0)).norm(), 141.42, 0.01);
}

TEST(RoadGraph, DisconnectedRoadsHaveNoRoute) {
    RoadGraph graph;
    addStraightRoad(graph, 0.0, 300.0, 1);
    addStraightRoad(graph, 10.0, 300.0, 2);

    const auto on_first = graph.candidatesNear(Vec3(150.0, 0.0, 0.0), 2.0, 1).front();
    const auto on_second = graph.candidatesNear(Vec3(150.0, 10.0, 0.0), 2.0, 1).front();
    EXPECT_LT(graph.routeDistance(on_first, on_second, 2000.0), 0.0);
}

TEST(RoadGraph, RouteDistanceRespectsItsBudget) {
    RoadGraph graph;
    addStraightRoad(graph, 0.0, 1000.0, 1);
    const auto from = graph.candidatesNear(Vec3(0.0, 0.0, 0.0), 5.0, 1).front();
    const auto to = graph.candidatesNear(Vec3(1000.0, 0.0, 0.0), 5.0, 1).front();

    EXPECT_NEAR(graph.routeDistance(from, to, 2000.0), 1000.0, 1e-6);
    EXPECT_LT(graph.routeDistance(from, to, 500.0), 0.0);
}

// --- The HMM ---------------------------------------------------------------

TEST(HmmMapMatcher, SnapsANoisyTrackToTheRoadItIsOn) {
    RoadGraph graph;
    addStraightRoad(graph, 0.0, 1000.0, 42);
    HmmMapMatcher matcher(graph, MapMatchParams{});

    std::mt19937 rng(20260930);
    std::normal_distribution<double> noise(0.0, 4.0);
    for (double north = 0.0; north <= 900.0; north += 50.0) {
        const auto result = matcher.step(Vec3(north, noise(rng), 0.0));
        ASSERT_TRUE(result.matched) << "at north=" << north;
        EXPECT_EQ(result.way_id, 42);
        EXPECT_NEAR(result.snapped_position.y(), 0.0, 1e-9);
        EXPECT_NEAR(result.snapped_position.x(), north, 1e-6);
        // Road runs due North, so the direction must too.
        EXPECT_NEAR(std::abs(result.segment_direction.x()), 1.0, 1e-9);
    }
}

TEST(HmmMapMatcher, StaysOnOneCarriagewayWhereNearestRoadWouldFlip) {
    // THE case that justifies the HMM over nearest-road snapping. Two parallel
    // carriageways 10 m apart, not connected to each other. The vehicle is on
    // the western one, but position noise regularly puts an observation closer
    // to the eastern one. Nearest-road flips carriageway whenever that happens;
    // the HMM cannot, because there is no route between them, so every
    // flip-flop transition has probability zero.
    RoadGraph graph;
    addStraightRoad(graph, 0.0, 1000.0, 100);   // western: the true one
    addStraightRoad(graph, 10.0, 1000.0, 200);  // eastern

    std::mt19937 rng(20260931);
    std::normal_distribution<double> noise(0.0, 4.5);

    std::vector<Vec3> observations;
    for (double north = 0.0; north <= 900.0; north += 25.0) {
        observations.emplace_back(north, noise(rng), 0.0);
    }

    // How often would naive nearest-road snapping have picked the wrong one?
    int nearest_road_flips = 0;
    for (const auto& observation : observations) {
        const auto nearest = graph.candidatesNear(observation, 50.0, 1);
        if (!nearest.empty() &&
            graph.segments()[nearest.front().segment_id].way_id != 100) {
            ++nearest_road_flips;
        }
    }
    ASSERT_GT(nearest_road_flips, 3)
        << "scenario is not actually ambiguous; nothing is being tested";
    std::cout << "[ INFO     ] nearest-road would pick the wrong carriageway "
              << nearest_road_flips << " of " << observations.size() << " times\n";

    HmmMapMatcher matcher(graph, MapMatchParams{});
    int hmm_flips = 0;
    for (const auto& observation : observations) {
        const auto result = matcher.step(observation);
        ASSERT_TRUE(result.matched);
        if (result.way_id != 100) ++hmm_flips;
    }
    std::cout << "[ INFO     ] HMM picked the wrong carriageway " << hmm_flips
              << " times\n";
    EXPECT_LT(hmm_flips, nearest_road_flips);
    EXPECT_LE(hmm_flips, 1) << "at most the very first, before any transition exists";
}

TEST(HmmMapMatcher, OfflineViterbiIsAtLeastAsGoodAsCausalDecoding) {
    // The offline pass can revise an early ambiguous match once later evidence
    // arrives; the causal one cannot. It must never be worse.
    RoadGraph graph;
    addStraightRoad(graph, 0.0, 1000.0, 100);
    addStraightRoad(graph, 10.0, 1000.0, 200);

    std::mt19937 rng(20260932);
    std::normal_distribution<double> noise(0.0, 4.5);
    std::vector<Vec3> observations;
    for (double north = 0.0; north <= 900.0; north += 25.0) {
        observations.emplace_back(north, noise(rng), 0.0);
    }

    HmmMapMatcher causal(graph, MapMatchParams{});
    int causal_errors = 0;
    for (const auto& observation : observations) {
        if (causal.step(observation).way_id != 100) ++causal_errors;
    }

    const HmmMapMatcher offline(graph, MapMatchParams{});
    int offline_errors = 0;
    for (const auto& result : offline.match(observations)) {
        if (!result.matched || result.way_id != 100) ++offline_errors;
    }

    std::cout << "[ INFO     ] causal errors: " << causal_errors
              << ", offline Viterbi errors: " << offline_errors << "\n";
    EXPECT_LE(offline_errors, causal_errors);
}

TEST(HmmMapMatcher, ReportsNoMatchWhenOffTheNetwork) {
    RoadGraph graph;
    addStraightRoad(graph, 0.0, 500.0, 1);
    HmmMapMatcher matcher(graph, MapMatchParams{});
    EXPECT_FALSE(matcher.step(Vec3(200.0, 5000.0, 0.0)).matched);
}

TEST(HmmMapMatcher, PrefersTheConnectedRouteOverACloserDisconnectedOne) {
    // Emission alone would pick the nearer road. The transition term should
    // override it, because reaching the nearer road implies a jump with no
    // route behind it.
    RoadGraph graph;
    addStraightRoad(graph, 0.0, 600.0, 100);
    addStraightRoad(graph, 6.0, 600.0, 200);
    HmmMapMatcher matcher(graph, MapMatchParams{});

    // Establish a confident history on the western carriageway.
    for (double north = 0.0; north <= 300.0; north += 50.0) {
        ASSERT_EQ(matcher.step(Vec3(north, 0.0, 0.0)).way_id, 100);
    }
    // Now one observation that is genuinely closer to the eastern one.
    const auto result = matcher.step(Vec3(350.0, 4.5, 0.0));
    EXPECT_EQ(result.way_id, 100) << "a single close observation must not "
                                     "override an impossible transition";
}

}  // namespace
