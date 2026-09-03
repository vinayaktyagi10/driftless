package com.driftless.ui

import android.graphics.Color
import com.driftless.fusion.FusedPosition
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.CopyrightOverlay
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline

/**
 * Owns everything drawn on the map: the current-position marker and the two
 * tracks.
 *
 * **Two tracks from the start, not one.** The aided track is what the engine
 * produces while GNSS is correcting it; the dead-reckoned track is what it
 * produces once it is coasting alone. Their divergence during an outage is the
 * entire visual argument of the demo.
 *
 * Since step 5 this consumes [FusedPosition] — the engine's output, already in
 * lat/lon — and no longer sees NED metres or the local-tangent frame at all.
 * That keeps exactly one place in the app where NED becomes geodetic, and it is
 * inside the engine, so the map cannot drift out of agreement with the filter
 * by converting differently.
 */
class TrackRenderer(private val map: MapView) {

    private val aided = Polyline(map).apply {
        outlinePaint.color = Color.parseColor("#2E7DF7")
        outlinePaint.strokeWidth = 8f
    }

    private val deadReckoned = Polyline(map).apply {
        outlinePaint.color = Color.parseColor("#F2A93B")
        outlinePaint.strokeWidth = 8f
    }

    private val marker = Marker(map).apply {
        setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        title = "Current position"
    }

    /**
     * False once the user pans or zooms. Auto-centring that fights the user is
     * worse than no auto-centring, and on stage the map will be driven by hand
     * as often as by the vehicle.
     */
    var followEnabled = true

    init {
        map.setTileSource(TileSourceFactory.MAPNIK)
        map.setMultiTouchControls(true)
        // Required by the OSM tile usage policy, and the map is on screen in
        // front of judges, so it is not optional in either sense.
        map.overlays.add(CopyrightOverlay(map.context))
        map.overlays.add(aided)
        map.overlays.add(deadReckoned)
        map.overlays.add(marker)
        map.controller.setZoom(DEFAULT_ZOOM)
    }

    private var lastDrawn: GeoPoint? = null
    private var lastCoasting: Boolean? = null

    /**
     * Moves the marker to the fused position and extends whichever track the
     * engine is currently feeding.
     *
     * Which track is chosen by [FusedPosition.confidence], which is the field
     * the contract set aside for exactly this. It holds at 1.0 while the
     * position is no worse than the fix it came from and decays once the engine
     * is coasting beyond that, so during normal 1 Hz driving everything lands on
     * [aided] and the orange line appears only during a real outage.
     *
     * Points closer than [MIN_SEGMENT_M] to the previous one are dropped from
     * the polyline, because a parked vehicle still produces a position ten times
     * a second and each lands a metre or two from the last — without this the
     * track accumulates a dense scribble wherever the vehicle stopped, which is
     * exactly where a judge looks to see whether the filter is drifting. The
     * marker still follows every update; only the drawn line is thinned.
     */
    fun addFusedPoint(fused: FusedPosition) {
        val point = GeoPoint(fused.lat, fused.lon)
        val coasting = fused.confidence < 1f
        val line = if (coasting) deadReckoned else aided

        val handover = lastCoasting != null && lastCoasting != coasting
        if (handover) {
            // Bridge the two polylines at the changeover, or the orange track
            // appears to begin somewhere the blue one never reached and the
            // divergence on screen reads as a jump rather than a departure.
            lastDrawn?.let { line.addPoint(it) }
        }

        val previous = lastDrawn
        if (handover || previous == null ||
            previous.distanceToAsDouble(point) >= MIN_SEGMENT_M
        ) {
            line.addPoint(point)
            lastDrawn = point
        }
        lastCoasting = coasting

        marker.position = point
        marker.rotation = -fused.headingDegrees
        // Dimmed while coasting, never invisible: the marker fading is the
        // cheapest honest signal that the position is no longer measured.
        marker.alpha = MARKER_ALPHA_FLOOR +
            (1f - MARKER_ALPHA_FLOOR) * fused.confidence.coerceIn(0f, 1f)

        if (followEnabled) {
            // setCenter, not animateTo. The engine republishes at 10 Hz and
            // animateTo spans ~300 ms, so animating would stack three
            // overlapping animations per position and the marker would lag the
            // track it is drawing. At 10 Hz the discrete steps *are* the smooth
            // motion.
            map.controller.setCenter(point)
        }
        map.invalidate()
    }

    fun recentre() {
        followEnabled = true
        marker.position?.let { map.controller.animateTo(it) }
    }

    fun clear() {
        lastDrawn = null
        lastCoasting = null
        aided.setPoints(emptyList())
        deadReckoned.setPoints(emptyList())
        map.invalidate()
    }

    private companion object {
        const val DEFAULT_ZOOM = 18.0

        /**
         * Metres of horizontal movement before a new vertex is drawn. Below the
         * ~2.75 m sigma_h observed on this handset, so real motion is never
         * thinned, but above the jitter of a stationary receiver.
         */
        const val MIN_SEGMENT_M = 2.0

        /** How faint a fully-unconfident marker gets. */
        const val MARKER_ALPHA_FLOOR = 0.35f
    }
}
