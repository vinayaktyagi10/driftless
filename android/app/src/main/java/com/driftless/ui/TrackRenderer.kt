package com.driftless.ui

import android.graphics.Color
import com.driftless.frames.LocalTangentFrame
import com.driftless.fusion.Position
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
 * **Two tracks from the start, not one.** The aided track is what the filter
 * produces with GNSS corrections; the dead-reckoned track is what it produces
 * without them. Their divergence during a simulated outage is the entire visual
 * argument of the demo, so the render path is shaped for it now — retrofitting a
 * second overlay during the final 48 hours means editing the one piece of code
 * that is on screen in front of judges.
 *
 * Step 3 feeds only [aided], from raw GNSS. Step 5 swaps in the fusion engine
 * and starts feeding [deadReckoned]; nothing here changes when it does.
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

    private var lastAided: Position? = null

    /**
     * Appends to the GNSS-aided track and moves the marker.
     *
     * Points closer than [MIN_SEGMENT_M] to the previous one are dropped from
     * the polyline. A parked vehicle still produces a fix every second, and each
     * one lands a metre or two away from the last, so without this the track
     * accumulates a dense scribble wherever the vehicle stopped — which is
     * exactly where a judge looks to see whether the filter is drifting. The
     * marker still follows every fix; only the drawn line is thinned.
     */
    fun addAidedPoint(frame: LocalTangentFrame, position: Position) {
        val point = frame.toGeoPoint(position)
        val previous = lastAided
        if (previous == null || previous.distanceTo(position) >= MIN_SEGMENT_M) {
            aided.addPoint(point)
            lastAided = position
        }
        marker.position = point
        if (followEnabled) {
            map.controller.animateTo(point)
        }
        map.invalidate()
    }

    /** Appends to the dead-reckoning-only track. Unused until step 5. */
    fun addDeadReckonedPoint(frame: LocalTangentFrame, position: Position) {
        deadReckoned.addPoint(frame.toGeoPoint(position))
        map.invalidate()
    }

    fun recentre() {
        followEnabled = true
        marker.position?.let { map.controller.animateTo(it) }
    }

    fun clear() {
        lastAided = null
        aided.setPoints(emptyList())
        deadReckoned.setPoints(emptyList())
        map.invalidate()
    }

    /**
     * The single place NED metres become map coordinates. Routed through
     * [LocalTangentFrame.toGeodetic] rather than given its own conversion, so
     * the map consumes exactly what the fusion engine emits and inherits the
     * round trip the frame tests already pin.
     */
    private fun LocalTangentFrame.toGeoPoint(position: Position): GeoPoint {
        val g = toGeodetic(position)
        return GeoPoint(g.latDeg, g.lonDeg)
    }

    private fun Position.distanceTo(other: Position): Double {
        val dn = north - other.north
        val de = east - other.east
        return kotlin.math.sqrt(dn * dn + de * de)
    }

    private companion object {
        const val DEFAULT_ZOOM = 18.0

        /**
         * Metres of horizontal movement before a new vertex is drawn. Below the
         * ~2.6 m sigma_h observed on this handset, so real motion is never
         * thinned, but above the jitter of a stationary receiver.
         */
        const val MIN_SEGMENT_M = 2.0
    }
}
