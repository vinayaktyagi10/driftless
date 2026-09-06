package com.driftless.mapmatch

import com.driftless.frames.LocalTangentFrame
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OsmReaderTest {

    @Test
    fun testParseMinimalOsmXml() {
        val xml = """
            <?xml version="1.0" encoding="UTF-8"?>
            <osm version="0.6">
                <node id="1" lat="26.8467" lon="75.8056"/>
                <node id="2" lat="26.8477" lon="75.8056"/>
                <node id="3" lat="26.8487" lon="75.8056"/>
                <node id="4" lat="26.8497" lon="75.8056"/>
                <way id="101">
                    <nd ref="1"/>
                    <nd ref="2"/>
                    <nd ref="3"/>
                    <tag k="highway" v="primary"/>
                    <tag k="name" v="Main Road"/>
                </way>
                <way id="102">
                    <nd ref="3"/>
                    <nd ref="4"/>
                    <tag k="highway" v="footway"/>
                </way>
            </osm>
        """.trimIndent()

        val frame = LocalTangentFrame.anchoredAt(26.8467, 75.8056, 0.0)
        val network = OsmReader.loadOsmXml(xml, frame)

        assertEquals(4, network.nodesLoaded)
        // Way 101 is primary (drivable), Way 102 is footway (skipped)
        assertEquals(1, network.waysLoaded)
        assertEquals(1, network.waysSkippedNotDrivable)
        assertEquals(2, network.graph.segmentCount)

        val seg0 = network.graph.segments[0]
        val seg1 = network.graph.segments[1]
        assertEquals(101, seg0.wayId)
        assertEquals(101, seg1.wayId)
        assertTrue(seg0.length() > 50.0)
    }

    @Test
    fun testAutoOffsetWhenOutsideBounds() {
        // Road at 26.8467, 75.8056. Anchor is 3.3 km away at 26.8644, 75.7794.
        val xml = """
            <?xml version="1.0" encoding="UTF-8"?>
            <osm version="0.6">
                <node id="1" lat="26.8467" lon="75.8056"/>
                <node id="2" lat="26.8477" lon="75.8056"/>
                <way id="101">
                    <nd ref="1"/>
                    <nd ref="2"/>
                    <tag k="highway" v="residential"/>
                </way>
            </osm>
        """.trimIndent()

        val farFrame = LocalTangentFrame.anchoredAt(26.8644, 75.7794, 0.0)
        val network = OsmReader.loadOsmXml(xml, farFrame)

        assertEquals(2, network.nodesLoaded)
        assertEquals(1, network.waysLoaded)
        // Verify segments are centered around the anchor in local NED (near 0, 0)
        val candidates = network.graph.candidatesNear(
            position = com.driftless.fusion.Vec3(0.0, 0.0, 0.0),
            radiusM = 60.0,
            maxCandidates = 5,
        )
        assertTrue("Road segments should be centered near anchor (0,0)", candidates.isNotEmpty())
    }

    @Test
    fun testTestFixtureAlwaysAutoOffsets() {
        val xml = """
            <?xml version="1.0" encoding="UTF-8"?>
            <osm version="0.6" generator="driftless-test-fixture">
                <node id="1" lat="26.8467" lon="75.8056"/>
                <node id="2" lat="26.8477" lon="75.8056"/>
                <way id="101">
                    <nd ref="1"/>
                    <nd ref="2"/>
                    <tag k="highway" v="residential"/>
                </way>
            </osm>
        """.trimIndent()

        val frame = LocalTangentFrame.anchoredAt(26.8467, 75.8056, 0.0)
        val network = OsmReader.loadOsmXml(xml, frame)
        val candidates = network.graph.candidatesNear(
            position = com.driftless.fusion.Vec3(0.0, 0.0, 0.0),
            radiusM = 60.0,
            maxCandidates = 5,
        )
        assertTrue("Test fixture should be centered at frame origin", candidates.isNotEmpty())
    }
}
