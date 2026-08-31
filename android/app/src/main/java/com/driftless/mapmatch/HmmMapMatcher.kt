package com.driftless.mapmatch

/**
 * Hidden Markov Map Matching (Newson & Krumm) against an offline OSM
 * road graph. States are candidate road segments near the fused
 * position; emission probability is distance to segment, transition
 * probability is route plausibility between consecutive candidates.
 * Also applies the non-holonomic constraint (no lateral slip) before
 * handing a snapped position back to the UKF as a correction.
 */
class HmmMapMatcher {
    // TODO: load offline OSM extract, viterbi over candidate states per fix
}
