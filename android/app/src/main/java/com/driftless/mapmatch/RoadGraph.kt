package com.driftless.mapmatch

import com.driftless.fusion.Vec3
import com.driftless.math.dot
import com.driftless.math.minus
import com.driftless.math.norm
import com.driftless.math.plus
import com.driftless.math.squaredNorm
import com.driftless.math.times
import java.util.PriorityQueue
import kotlin.math.abs
import kotlin.math.hypot

data class RoadSegment(
    val id: Int = -1,
    val wayId: Int = -1,
    val startNode: Int = -1,
    val endNode: Int = -1,
    val start: Vec3 = Vec3(),
    val end: Vec3 = Vec3(),
) {
    fun length(): Double = (end - start).norm()

    fun direction(): Vec3 {
        val delta = end - start
        val n = delta.norm()
        return if (n <= 0.0) Vec3(1.0, 0.0, 0.0) else delta * (1.0 / n)
    }
}

data class SegmentProjection(
    val segmentId: Int = -1,
    val point: Vec3 = Vec3(),
    val distanceM: Double = 0.0,
    val alongM: Double = 0.0,
)

/**
 * Road network graph in local NED coordinates.
 */
class RoadGraph {

    private val nodesList = mutableListOf<Vec3>()
    private val segmentsList = mutableListOf<RoadSegment>()

    val nodes: List<Vec3> get() = nodesList
    val segments: List<RoadSegment> get() = segmentsList
    val segmentCount: Int get() = segmentsList.size

    private data class AdjacentNode(val node: Int, val length: Double)

    private var adjacency: Array<MutableList<AdjacentNode>>? = null

    fun addNode(nedPosition: Vec3): Int {
        nodesList.add(nedPosition)
        adjacency = null
        return nodesList.size - 1
    }

    fun addSegment(startNode: Int, endNode: Int, wayId: Int): Int {
        if (startNode < 0 || endNode < 0 || startNode >= nodesList.size || endNode >= nodesList.size) {
            return -1
        }
        val pStart = nodesList[startNode]
        val pEnd = nodesList[endNode]
        val seg = RoadSegment(
            id = segmentsList.size,
            wayId = wayId,
            startNode = startNode,
            endNode = endNode,
            start = pStart,
            end = pEnd,
        )
        if (seg.length() <= 0.0) return -1

        segmentsList.add(seg)
        adjacency = null
        return seg.id
    }

    private fun ensureAdjacency() {
        if (adjacency != null) return
        val adj = Array(nodesList.size) { mutableListOf<AdjacentNode>() }
        for (seg in segmentsList) {
            val len = seg.length()
            adj[seg.startNode].add(AdjacentNode(seg.endNode, len))
            adj[seg.endNode].add(AdjacentNode(seg.startNode, len))
        }
        adjacency = adj
    }

    fun candidatesNear(
        position: Vec3,
        radiusM: Double,
        maxCandidates: Int,
    ): List<SegmentProjection> {
        val candidates = mutableListOf<SegmentProjection>()
        for (seg in segmentsList) {
            val proj = project(seg, position)
            if (proj.distanceM <= radiusM) {
                candidates.add(proj)
            }
        }
        candidates.sortBy { it.distanceM }
        return if (candidates.size > maxCandidates) candidates.subList(0, maxCandidates) else candidates
    }

    fun routeDistance(
        from: SegmentProjection,
        to: SegmentProjection,
        maxDistanceM: Double,
    ): Double {
        if (from.segmentId < 0 || to.segmentId < 0) return -1.0
        if (from.segmentId >= segmentsList.size || to.segmentId >= segmentsList.size) return -1.0

        val fromSeg = segmentsList[from.segmentId]
        val toSeg = segmentsList[to.segmentId]

        if (from.segmentId == to.segmentId) {
            return abs(to.alongM - from.alongM)
        }

        ensureAdjacency()
        val adj = adjacency ?: return -1.0

        val numNodes = nodesList.size
        val distances = DoubleArray(numNodes) { Double.POSITIVE_INFINITY }

        data class Entry(val cost: Double, val node: Int) : Comparable<Entry> {
            override fun compareTo(other: Entry): Int = cost.compareTo(other.cost)
        }

        val frontier = PriorityQueue<Entry>()

        fun seed(node: Int, cost: Double) {
            if (cost < distances[node]) {
                distances[node] = cost
                frontier.add(Entry(cost, node))
            }
        }

        seed(fromSeg.startNode, from.alongM)
        seed(fromSeg.endNode, fromSeg.length() - from.alongM)

        val toStartCost = to.alongM
        val toEndCost = toSeg.length() - to.alongM

        var best = Double.POSITIVE_INFINITY

        while (frontier.isNotEmpty()) {
            val entry = frontier.poll() ?: break
            val cost = entry.cost
            val node = entry.node
            if (cost > distances[node]) continue
            if (cost > maxDistanceM || cost >= best) break

            if (node == toSeg.startNode) {
                best = minOf(best, cost + toStartCost)
            }
            if (node == toSeg.endNode) {
                best = minOf(best, cost + toEndCost)
            }

            for (neighbour in adj[node]) {
                val nextCost = cost + neighbour.length
                if (nextCost < distances[neighbour.node]) {
                    distances[neighbour.node] = nextCost
                    frontier.add(Entry(nextCost, neighbour.node))
                }
            }
        }

        if (!best.isFinite() || best > maxDistanceM) return -1.0
        return best
    }

    companion object {
        fun project(segment: RoadSegment, position: Vec3): SegmentProjection {
            val delta = segment.end - segment.start
            val lengthSq = delta.squaredNorm()

            var t = 0.0
            if (lengthSq > 0.0) {
                t = ((position - segment.start).dot(delta) / lengthSq).coerceIn(0.0, 1.0)
            }

            val point = segment.start + (delta * t)
            val alongM = t * segment.length()
            val offset = position - point
            val distanceM = hypot(offset.x, offset.y)

            return SegmentProjection(
                segmentId = segment.id,
                point = point,
                distanceM = distanceM,
                alongM = alongM,
            )
        }
    }
}
