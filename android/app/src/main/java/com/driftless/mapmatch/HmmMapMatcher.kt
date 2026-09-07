package com.driftless.mapmatch

import com.driftless.fusion.MapMatchResult
import com.driftless.fusion.Vec3
import com.driftless.math.minus
import com.driftless.math.norm
import kotlin.math.abs

data class HmmParams(
    val candidateRadiusM: Double = 50.0,
    val maxCandidatesPerObservation: Int = 8,
    val emissionSigmaM: Double = 10.0,
    val transitionBetaM: Double = 5.0,
    val maxRouteDistanceM: Double = 2000.0,
)

/**
 * Hidden Markov Map Matching (Newson & Krumm 2009) against an offline road graph.
 *
 * Implements online causal forward decoding for real-time filter updates [step]
 * and offline full Viterbi with backtracking [match].
 */
class HmmMapMatcher(
    private val graph: RoadGraph,
    private val params: HmmParams = HmmParams(),
) {

    private data class Hypothesis(
        val projection: SegmentProjection,
        var logProbability: Double,
    )

    private var hypotheses = mutableListOf<Hypothesis>()
    private var lastPosition: Vec3 = Vec3()
    private var hasLastPosition = false

    fun reset() {
        hypotheses.clear()
        hasLastPosition = false
    }

    fun emissionLogProbability(distanceM: Double): Double {
        val scaled = distanceM / params.emissionSigmaM
        return -0.5 * scaled * scaled
    }

    fun transitionLogProbability(
        from: SegmentProjection,
        to: SegmentProjection,
        straightLineM: Double,
    ): Double {
        val routeM = graph.routeDistance(from, to, params.maxRouteDistanceM)
        if (routeM < 0.0) return Double.NEGATIVE_INFINITY
        return -abs(straightLineM - routeM) / params.transitionBetaM
    }

    private fun toResult(projection: SegmentProjection): MapMatchResult {
        if (projection.segmentId < 0 || projection.segmentId >= graph.segments.size) {
            return MapMatchResult()
        }
        val seg = graph.segments[projection.segmentId]
        return MapMatchResult(
            matched = true,
            segmentId = projection.segmentId,
            wayId = seg.wayId,
            snappedPosition = projection.point,
            segmentDirection = seg.direction(),
            distanceToRoadM = projection.distanceM,
        )
    }

    /**
     * Causal forward decoding, one observation at a time.
     */
    fun step(position: Vec3): MapMatchResult {
        val candidates = graph.candidatesNear(
            position = position,
            radiusM = params.candidateRadiusM,
            maxCandidates = params.maxCandidatesPerObservation,
        )

        if (candidates.isEmpty()) {
            reset()
            return MapMatchResult()
        }

        val next = mutableListOf<Hypothesis>()

        if (hypotheses.isEmpty() || !hasLastPosition) {
            for (cand in candidates) {
                next.add(Hypothesis(cand, emissionLogProbability(cand.distanceM)))
            }
        } else {
            val straightLine = (position - lastPosition).norm()
            for (cand in candidates) {
                var best = Double.NEGATIVE_INFINITY
                for (prev in hypotheses) {
                    if (prev.logProbability == Double.NEGATIVE_INFINITY) continue
                    val trans = transitionLogProbability(prev.projection, cand, straightLine)
                    if (trans == Double.NEGATIVE_INFINITY) continue
                    val score = prev.logProbability + trans
                    if (score > best) {
                        best = score
                    }
                }
                if (best != Double.NEGATIVE_INFINITY) {
                    next.add(Hypothesis(cand, best + emissionLogProbability(cand.distanceM)))
                }
            }

            if (next.isEmpty()) {
                // Every transition was impossible; restart from emission alone
                for (cand in candidates) {
                    next.add(Hypothesis(cand, emissionLogProbability(cand.distanceM)))
                }
            }
        }

        // Renormalize by subtracting max score
        val bestScore = next.maxOf { it.logProbability }
        if (bestScore.isFinite()) {
            for (hyp in next) {
                hyp.logProbability -= bestScore
            }
        }

        hypotheses = next
        lastPosition = position
        hasLastPosition = true

        val bestHypothesis = hypotheses.maxByOrNull { it.logProbability }
        return if (bestHypothesis != null) toResult(bestHypothesis.projection) else MapMatchResult()
    }

    /**
     * Full offline Viterbi with backtracking over a whole trajectory.
     */
    fun match(positions: List<Vec3>): List<MapMatchResult> {
        val results = MutableList(positions.size) { MapMatchResult() }
        if (positions.isEmpty()) return results

        val candidates = Array(positions.size) { t ->
            graph.candidatesNear(positions[t], params.candidateRadiusM, params.maxCandidatesPerObservation)
        }
        val scores = Array(positions.size) { t ->
            DoubleArray(candidates[t].size) { Double.NEGATIVE_INFINITY }
        }
        val backpointers = Array(positions.size) { t ->
            IntArray(candidates[t].size) { -1 }
        }

        // Forward pass
        var first = positions.size
        for (t in positions.indices) {
            if (candidates[t].isEmpty()) continue
            if (first == positions.size) {
                for (i in candidates[t].indices) {
                    scores[t][i] = emissionLogProbability(candidates[t][i].distanceM)
                }
                first = t
                continue
            }

            var previous = t - 1
            while (previous > first && candidates[previous].isEmpty()) previous--

            val straightLine = (positions[t] - positions[previous]).norm()
            for (i in candidates[t].indices) {
                var best = Double.NEGATIVE_INFINITY
                var bestIdx = -1
                for (j in candidates[previous].indices) {
                    if (scores[previous][j] == Double.NEGATIVE_INFINITY) continue
                    val trans = transitionLogProbability(candidates[previous][j], candidates[t][i], straightLine)
                    if (trans == Double.NEGATIVE_INFINITY) continue
                    val total = scores[previous][j] + trans
                    if (total > best) {
                        best = total
                        bestIdx = j
                    }
                }
                if (best == Double.NEGATIVE_INFINITY) {
                    scores[t][i] = emissionLogProbability(candidates[t][i].distanceM)
                    backpointers[t][i] = -1
                } else {
                    scores[t][i] = best + emissionLogProbability(candidates[t][i].distanceM)
                    backpointers[t][i] = bestIdx
                }
            }

            val bestScore = scores[t].maxOrNull() ?: Double.NEGATIVE_INFINITY
            if (bestScore.isFinite()) {
                for (i in scores[t].indices) {
                    scores[t][i] -= bestScore
                }
            }
        }

        if (first == positions.size) return results

        // Backward pass
        var last = positions.size - 1
        while (last > first && candidates[last].isEmpty()) last--

        var bestLastIdx = 0
        var bestLastScore = Double.NEGATIVE_INFINITY
        for (i in scores[last].indices) {
            if (scores[last][i] > bestLastScore) {
                bestLastScore = scores[last][i]
                bestLastIdx = i
            }
        }

        var index = bestLastIdx
        for (t in last downTo first) {
            if (candidates[t].isEmpty() || index < 0) continue
            results[t] = toResult(candidates[t][index])
            val nextIndex = backpointers[t][index]
            index = if (nextIndex >= 0) {
                nextIndex
            } else {
                var prevT = t - 1
                while (prevT >= first && candidates[prevT].isEmpty()) prevT--
                if (prevT >= first && candidates[prevT].isNotEmpty()) {
                    var bestP = 0
                    var bestPScore = Double.NEGATIVE_INFINITY
                    for (j in scores[prevT].indices) {
                        if (scores[prevT][j] > bestPScore) {
                            bestPScore = scores[prevT][j]
                            bestP = j
                        }
                    }
                    bestP
                } else {
                    -1
                }
            }
        }

        return results
    }
}
