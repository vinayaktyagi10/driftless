#include "driftless/osm_reader.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace driftless::osm {
namespace {

constexpr std::string_view kDrivable[] = {
    "motorway",      "motorway_link", "trunk",        "trunk_link",
    "primary",       "primary_link",  "secondary",    "secondary_link",
    "tertiary",      "tertiary_link", "unclassified", "residential",
    "living_street", "service",       "road",
};

// Value of attribute `name` within the element text `element`, or nullopt.
std::optional<std::string_view> attribute(std::string_view element,
                                          std::string_view name) {
    std::string pattern;
    pattern.reserve(name.size() + 2);
    pattern += name;
    pattern += "=\"";
    const std::size_t key = element.find(pattern);
    if (key == std::string_view::npos) return std::nullopt;
    const std::size_t start = key + pattern.size();
    const std::size_t end = element.find('"', start);
    if (end == std::string_view::npos) return std::nullopt;
    return element.substr(start, end - start);
}

std::optional<double> toDouble(std::string_view text) {
    // from_chars for double is the only locale-independent option that does not
    // allocate; an OSM file is always '.'-decimal regardless of system locale,
    // which is exactly the bug stod would introduce.
    double value = 0.0;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{}) return std::nullopt;
    return value;
}

std::optional<long long> toLongLong(std::string_view text) {
    long long value = 0;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{}) return std::nullopt;
    return value;
}

}  // namespace

bool isDrivableHighway(std::string_view highway_value) {
    return std::find(std::begin(kDrivable), std::end(kDrivable), highway_value) !=
           std::end(kDrivable);
}

bool loadOsmXml(std::string_view xml, OsmRoadNetwork& out, std::string* error,
                std::optional<GeoPoint> origin) {
    const auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };

    // Pass 1: every node's coordinates, and the bounding box for the origin.
    std::unordered_map<long long, GeoPoint> raw_nodes;
    double min_lat = 90.0, max_lat = -90.0, min_lon = 180.0, max_lon = -180.0;

    for (std::size_t cursor = xml.find("<node"); cursor != std::string_view::npos;
         cursor = xml.find("<node", cursor + 5)) {
        const std::size_t end = xml.find('>', cursor);
        if (end == std::string_view::npos) break;
        const std::string_view element = xml.substr(cursor, end - cursor);

        const auto id = attribute(element, "id");
        const auto lat = attribute(element, "lat");
        const auto lon = attribute(element, "lon");
        if (!id || !lat || !lon) continue;

        const auto id_value = toLongLong(*id);
        const auto lat_value = toDouble(*lat);
        const auto lon_value = toDouble(*lon);
        if (!id_value || !lat_value || !lon_value) continue;

        raw_nodes[*id_value] = GeoPoint{*lat_value, *lon_value};
        min_lat = std::min(min_lat, *lat_value);
        max_lat = std::max(max_lat, *lat_value);
        min_lon = std::min(min_lon, *lon_value);
        max_lon = std::max(max_lon, *lon_value);
    }
    if (raw_nodes.empty()) return fail("no <node> elements found");

    out.nodes_loaded = raw_nodes.size();
    out.origin = origin.value_or(
        GeoPoint{0.5 * (min_lat + max_lat), 0.5 * (min_lon + max_lon)});
    const LocalTangentFrame frame(out.origin);

    // Pass 2: ways. Nodes are added to the graph lazily, so only those actually
    // used by a drivable way end up in it.
    std::unordered_map<long long, int> graph_node_of;
    const auto graphNode = [&](long long osm_id) -> int {
        const auto existing = graph_node_of.find(osm_id);
        if (existing != graph_node_of.end()) return existing->second;
        const auto coordinates = raw_nodes.find(osm_id);
        if (coordinates == raw_nodes.end()) return -1;
        const int index = out.graph.addNode(frame.toNed(coordinates->second));
        graph_node_of.emplace(osm_id, index);
        return index;
    };

    for (std::size_t cursor = xml.find("<way"); cursor != std::string_view::npos;
         cursor = xml.find("<way", cursor + 4)) {
        std::size_t close = xml.find("</way>", cursor);
        if (close == std::string_view::npos) close = xml.size();
        const std::string_view body = xml.substr(cursor, close - cursor);

        const auto way_id = toLongLong(attribute(body, "id").value_or("0"));

        // The highway tag decides whether this way is a road at all.
        std::string_view highway;
        for (std::size_t tag = body.find("<tag"); tag != std::string_view::npos;
             tag = body.find("<tag", tag + 4)) {
            const std::size_t tag_end = body.find('>', tag);
            if (tag_end == std::string_view::npos) break;
            const std::string_view element = body.substr(tag, tag_end - tag);
            if (attribute(element, "k").value_or("") == "highway") {
                highway = attribute(element, "v").value_or("");
                break;
            }
        }
        if (highway.empty()) continue;
        if (!isDrivableHighway(highway)) {
            ++out.ways_skipped_not_drivable;
            continue;
        }

        std::vector<int> way_nodes;
        for (std::size_t nd = body.find("<nd"); nd != std::string_view::npos;
             nd = body.find("<nd", nd + 3)) {
            const std::size_t nd_end = body.find('>', nd);
            if (nd_end == std::string_view::npos) break;
            const auto reference =
                attribute(body.substr(nd, nd_end - nd), "ref");
            if (!reference) continue;
            const auto reference_id = toLongLong(*reference);
            if (!reference_id) continue;
            const int index = graphNode(*reference_id);
            // A node outside the extract: skip it rather than abort. This can
            // split a way at the extract boundary, which is a known limitation.
            if (index >= 0) way_nodes.push_back(index);
        }
        if (way_nodes.size() < 2) continue;

        for (std::size_t i = 1; i < way_nodes.size(); ++i) {
            out.graph.addSegment(way_nodes[i - 1], way_nodes[i],
                                 static_cast<int>(way_id.value_or(0)));
        }
        ++out.ways_loaded;
    }

    if (out.graph.segmentCount() == 0) return fail("no drivable ways found");
    return true;
}

bool loadOsmFile(const std::string& path, OsmRoadNetwork& out, std::string* error,
                 std::optional<GeoPoint> origin) {
    std::ifstream stream(path);
    if (!stream) {
        if (error != nullptr) *error = "could not open " + path;
        return false;
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    const std::string contents = buffer.str();
    return loadOsmXml(contents, out, error, origin);
}

}  // namespace driftless::osm
