// The OSM reader, against a hand-written fixture whose contents are known
// exactly. Real extracts come as .pbf and are far messier; this pins the
// parsing rules, not robustness to the whole of OpenStreetMap.

#include "driftless/osm_reader.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using driftless::osm::isDrivableHighway;
using driftless::osm::loadOsmFile;
using driftless::osm::loadOsmXml;
using driftless::osm::OsmRoadNetwork;

// Set by CMake so the test can find its fixture regardless of build directory.
std::string fixturePath() { return std::string(DRIFTLESS_TEST_FIXTURE_DIR) + "/grid.osm"; }

TEST(OsmReader, LoadsTheGridFixture) {
    OsmRoadNetwork network;
    std::string error;
    ASSERT_TRUE(loadOsmFile(fixturePath(), network, &error)) << error;

    // 3x3 grid: 3 east-west ways and 3 north-south ways, each of 3 nodes, so
    // two segments apiece -- 12 segments in total.
    EXPECT_EQ(network.ways_loaded, 6u);
    EXPECT_EQ(network.graph.segmentCount(), 12u);
    EXPECT_EQ(network.graph.nodes().size(), 9u);
}

TEST(OsmReader, ExcludesFootwaysFromTheDrivableNetwork) {
    // A matcher that can snap a car onto a footpath will do exactly that in a
    // pedestrianised centre, so the filter has to be excluded at load time.
    OsmRoadNetwork network;
    ASSERT_TRUE(loadOsmFile(fixturePath(), network, nullptr));
    EXPECT_EQ(network.ways_skipped_not_drivable, 1u);

    // The footway's two nodes exist in the file but must not reach the graph.
    EXPECT_EQ(network.nodes_loaded, 11u) << "file really does contain 11 nodes";
    EXPECT_EQ(network.graph.nodes().size(), 9u) << "only the 9 grid nodes used";
}

TEST(OsmReader, DrivableClassification) {
    EXPECT_TRUE(isDrivableHighway("motorway"));
    EXPECT_TRUE(isDrivableHighway("residential"));
    EXPECT_TRUE(isDrivableHighway("secondary_link"));
    EXPECT_FALSE(isDrivableHighway("footway"));
    EXPECT_FALSE(isDrivableHighway("cycleway"));
    EXPECT_FALSE(isDrivableHighway("steps"));
    EXPECT_FALSE(isDrivableHighway(""));
}

TEST(OsmReader, GridGeometryHasTheExpectedSpacing) {
    OsmRoadNetwork network;
    ASSERT_TRUE(loadOsmFile(fixturePath(), network, nullptr));
    // The fixture uses 0.0009 deg latitude and 0.0010 deg longitude spacing,
    // both about 100 m at this latitude.
    for (const auto& segment : network.graph.segments()) {
        EXPECT_NEAR(segment.length(), 100.0, 5.0);
    }
}

TEST(OsmReader, HonoursAnExplicitOrigin) {
    OsmRoadNetwork network;
    const driftless::GeoPoint origin{26.8467, 75.8056};
    ASSERT_TRUE(loadOsmFile(fixturePath(), network, nullptr, origin));
    EXPECT_DOUBLE_EQ(network.origin.latitude_deg, origin.latitude_deg);
    // Node 1 sits exactly on that origin, so some graph node must be at zero.
    bool found_origin_node = false;
    for (const auto& node : network.graph.nodes()) {
        if (node.norm() < 1e-6) found_origin_node = true;
    }
    EXPECT_TRUE(found_origin_node);
}

TEST(OsmReader, ReportsFailureRatherThanReturningAnEmptyGraph) {
    OsmRoadNetwork network;
    std::string error;
    EXPECT_FALSE(loadOsmXml("<osm></osm>", network, &error));
    EXPECT_FALSE(error.empty());

    error.clear();
    EXPECT_FALSE(loadOsmFile("/nonexistent/path.osm", network, &error));
    EXPECT_FALSE(error.empty());

    // Nodes but no drivable ways is also a failure, not an empty success.
    error.clear();
    EXPECT_FALSE(loadOsmXml(
        R"(<osm><node id="1" lat="1.0" lon="2.0"/><node id="2" lat="1.001" lon="2.0"/>)"
        R"(<way id="9"><nd ref="1"/><nd ref="2"/><tag k="highway" v="footway"/></way></osm>)",
        network, &error));
    EXPECT_FALSE(error.empty());
}

TEST(OsmReader, IsNotFooledByLocaleOrAttributeOrdering) {
    // Attributes in a different order, and a decimal point that a locale-aware
    // parser could misread. Both are real things that appear in the wild.
    OsmRoadNetwork network;
    ASSERT_TRUE(loadOsmXml(
        R"(<osm><node lon="75.8056" id="1" lat="26.8467"/>)"
        R"(<node lat="26.8476" lon="75.8056" id="2"/>)"
        R"(<way id="7"><tag k="highway" v="primary"/><nd ref="1"/><nd ref="2"/></way></osm>)",
        network, nullptr));
    ASSERT_EQ(network.graph.segmentCount(), 1u);
    EXPECT_NEAR(network.graph.segments()[0].length(), 100.0, 5.0);
}

}  // namespace
