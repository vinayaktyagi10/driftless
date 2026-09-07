package com.driftless.mapmatch

import com.driftless.fusion.Vec3
import com.driftless.math.minus
import com.driftless.math.norm
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class HmmMapMatcherTest {

    private fun addStraightRoad(graph: RoadGraph, eastM: Double, lengthM: Double, wayId: Int): Int {
        var prev = graph.addNode(Vec3(0.0, eastM, 0.0))
        var north = 100.0
        while (north <= lengthM + 1e-9) {
            val next = graph.addNode(Vec3(north, eastM, 0.0))
            graph.addSegment(prev, next, wayId)
            prev = next
            north += 100.0
        }
        return wayId
    }

    @Test
    fun testRoadGraphProjectsPerpendicularlyAndClampsBeyondEnds() {
        val graph = RoadGraph()
        val a = graph.addNode(Vec3(0.0, 0.0, 0.0))
        val b = graph.addNode(Vec3(100.0, 0.0, 0.0))
        val id = graph.addSegment(a, b, 1)
        assertTrue(id >= 0)
        val segment = graph.segments[id]

        val middle = RoadGraph.project(segment, Vec3(50.0, 7.0, 0.0))
        assertEquals(50.0, middle.point.x, 1e-12)
        assertEquals(7.0, middle.distanceM, 1e-12)
        assertEquals(50.0, middle.alongM, 1e-12)

        // Past the far end: snaps to endpoint
        val beyond = RoadGraph.project(segment, Vec3(180.0, 0.0, 0.0))
        assertEquals(100.0, beyond.point.x, 1e-12)
        assertEquals(80.0, beyond.distanceM, 1e-12)
        assertEquals(100.0, beyond.alongM, 1e-12)
    }

    @Test
    fun testRoadGraphRejectsZeroLengthSegments() {
        val graph = RoadGraph()
        val a = graph.addNode(Vec3(5.0, 5.0, 0.0))
        val b = graph.addNode(Vec3(5.0, 5.0, 0.0))
        assertEquals(-1, graph.addSegment(a, b, 1))
        assertEquals(0, graph.segmentCount)
    }

    @Test
    fun testCandidateSearchRespectsRadiusOrderingAndCap() {
        val graph = RoadGraph()
        addStraightRoad(graph, 0.0, 300.0, 1)
        addStraightRoad(graph, 20.0, 300.0, 2)
        addStraightRoad(graph, 500.0, 300.0, 3)

        val candidates = graph.candidatesNear(Vec3(150.0, 2.0, 0.0), 50.0, 8)
        assertTrue(candidates.size >= 2)
        assertTrue(candidates[0].distanceM < candidates[1].distanceM)
        assertEquals(2.0, candidates[0].distanceM, 1e-9)
        for (c in candidates) {
            assertTrue(c.distanceM <= 50.0)
        }

        assertEquals(3, graph.candidatesNear(Vec3(150.0, 2.0, 0.0), 1000.0, 3).size)
    }

    @Test
    fun testRouteDistanceAlongASingleRoadIsArcLength() {
        val graph = RoadGraph()
        addStraightRoad(graph, 0.0, 500.0, 1)

        val from = graph.candidatesNear(Vec3(50.0, 0.0, 0.0), 10.0, 1).first()
        val to = graph.candidatesNear(Vec3(430.0, 0.0, 0.0), 10.0, 1).first()
        assertEquals(380.0, graph.routeDistance(from, to, 2000.0), 1e-6)
    }

    @Test
    fun testRouteDistanceGoesAroundCornersNotThroughThem() {
        val graph = RoadGraph()
        val corner = graph.addNode(Vec3(0.0, 0.0, 0.0))
        val north = graph.addNode(Vec3(100.0, 0.0, 0.0))
        val east = graph.addNode(Vec3(0.0, 100.0, 0.0))
        graph.addSegment(corner, north, 1)
        graph.addSegment(corner, east, 2)

        val from = graph.candidatesNear(Vec3(100.0, 0.0, 0.0), 5.0, 1).first()
        val to = graph.candidatesNear(Vec3(0.0, 100.0, 0.0), 5.0, 1).first()

        assertEquals(200.0, graph.routeDistance(from, to, 2000.0), 1e-6)
        assertEquals(141.42, (Vec3(100.0, 0.0, 0.0) - Vec3(0.0, 100.0, 0.0)).norm(), 0.01)
    }

    @Test
    fun testDisconnectedRoadsHaveNoRoute() {
        val graph = RoadGraph()
        addStraightRoad(graph, 0.0, 300.0, 1)
        addStraightRoad(graph, 10.0, 300.0, 2)

        val onFirst = graph.candidatesNear(Vec3(150.0, 0.0, 0.0), 2.0, 1).first()
        val onSecond = graph.candidatesNear(Vec3(150.0, 10.0, 0.0), 2.0, 1).first()
        assertTrue(graph.routeDistance(onFirst, onSecond, 2000.0) < 0.0)
    }

    @Test
    fun testHmmSnapsNoisyTrackToTheRoadItIsOn() {
        val graph = RoadGraph()
        addStraightRoad(graph, 0.0, 1000.0, 42)
        val matcher = HmmMapMatcher(graph, HmmParams())

        val rng = Random(20260930)
        var north = 0.0
        while (north <= 900.0) {
            val noisyY = rng.nextGaussian() * 4.0
            val result = matcher.step(Vec3(north, noisyY, 0.0))
            assertTrue(result.matched)
            assertEquals(42, result.wayId)
            assertEquals(0.0, result.snappedPosition.y, 1e-9)
            assertEquals(north, result.snappedPosition.x, 1e-6)
            north += 50.0
        }
    }

    @Test
    fun testHmmStaysOnOneCarriagewayWhereNearestRoadWouldFlip() {
        val graph = RoadGraph()
        addStraightRoad(graph, 0.0, 1000.0, 100)  // western true road
        addStraightRoad(graph, 10.0, 1000.0, 200) // eastern parallel road

        val rng = Random(42)
        val observations = mutableListOf<Vec3>()
        var north = 0.0
        while (north <= 900.0) {
            val noiseY = if (north == 0.0) 0.0 else rng.nextGaussian() * 4.5
            observations.add(Vec3(north, noiseY, 0.0))
            north += 25.0
        }

        var nearestRoadFlips = 0
        for (obs in observations) {
            val nearest = graph.candidatesNear(obs, 50.0, 1)
            if (nearest.isNotEmpty() && graph.segments[nearest.first().segmentId].wayId != 100) {
                nearestRoadFlips++
            }
        }
        assertTrue("Scenario must be ambiguous (nearest flips=$nearestRoadFlips)", nearestRoadFlips > 0)

        val matcher = HmmMapMatcher(graph, HmmParams())
        var hmmFlips = 0
        for (obs in observations) {
            val result = matcher.step(obs)
            assertTrue(result.matched)
            if (result.wayId != 100) hmmFlips++
        }

        println("nearestRoadFlips = $nearestRoadFlips, hmmFlips = $hmmFlips")
        assertTrue("HMM flips ($hmmFlips) must be much fewer than nearest road ($nearestRoadFlips)", hmmFlips <= nearestRoadFlips)
        assertTrue("HMM flips should be <= 1 (was $hmmFlips)", hmmFlips <= 1)
    }

    @Test
    fun testOfflineViterbiIsAtLeastAsGoodAsCausalDecoding() {
        val graph = RoadGraph()
        addStraightRoad(graph, 0.0, 1000.0, 100)
        addStraightRoad(graph, 10.0, 1000.0, 200)

        val rng = Random(20260932)
        val observations = mutableListOf<Vec3>()
        var north = 0.0
        while (north <= 900.0) {
            observations.add(Vec3(north, rng.nextGaussian() * 4.5, 0.0))
            north += 25.0
        }

        val causal = HmmMapMatcher(graph, HmmParams())
        var causalErrors = 0
        for (obs in observations) {
            if (causal.step(obs).wayId != 100) causalErrors++
        }

        val offline = HmmMapMatcher(graph, HmmParams())
        var offlineErrors = 0
        for (res in offline.match(observations)) {
            if (!res.matched || res.wayId != 100) offlineErrors++
        }

        assertTrue("Offline Viterbi errors ($offlineErrors) must be <= causal ($causalErrors)", offlineErrors <= causalErrors)
    }

    @Test
    fun testReportsNoMatchWhenOffTheNetwork() {
        val graph = RoadGraph()
        addStraightRoad(graph, 0.0, 500.0, 1)
        val matcher = HmmMapMatcher(graph, HmmParams())
        assertFalse(matcher.step(Vec3(200.0, 5000.0, 0.0)).matched)
    }
}
