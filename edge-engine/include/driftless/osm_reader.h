#pragma once

#include "driftless/road_graph.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace driftless::osm {

struct OsmRoadNetwork {
    RoadGraph graph;
    // Origin the graph's local NED coordinates are relative to. Defaults to the
    // centroid of the extract's bounding box unless one is supplied.
    GeoPoint origin;
    std::size_t nodes_loaded = 0;
    std::size_t ways_loaded = 0;
    std::size_t ways_skipped_not_drivable = 0;
};

// Minimal OSM XML reader: nodes, ways, and the `highway` tag. Deliberately not
// a general XML parser -- it does a linear scan for the four constructs that
// matter, which is enough for an .osm extract and avoids taking a dependency
// for one file format.
//
// Only drivable highway classes are kept. Footways, cycleways and steps are
// excluded, because a matcher that can snap a car onto a footpath will do
// exactly that in a pedestrianised city centre.
//
// LIMITATIONS, all of which matter on a real extract and none of which matter
// on the fixtures this is currently tested against:
//   - .osm XML only, not .pbf (which is what real extracts are distributed as).
//   - No `oneway` handling; RoadGraph routes both directions regardless.
//   - No turn restrictions.
//   - Way nodes outside the extract are silently dropped, which can split a
//     way into disconnected pieces at the boundary.
[[nodiscard]] bool loadOsmXml(std::string_view xml, OsmRoadNetwork& out,
                              std::string* error = nullptr,
                              std::optional<GeoPoint> origin = std::nullopt);

[[nodiscard]] bool loadOsmFile(const std::string& path, OsmRoadNetwork& out,
                               std::string* error = nullptr,
                               std::optional<GeoPoint> origin = std::nullopt);

// Whether a `highway=*` value is something a car may drive on.
[[nodiscard]] bool isDrivableHighway(std::string_view highway_value);

}  // namespace driftless::osm
