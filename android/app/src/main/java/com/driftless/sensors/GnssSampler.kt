package com.driftless.sensors

/**
 * FusedLocationProviderClient wrapper. Also surfaces GNSS status
 * (satellite count, signal strength) so the fusion engine can detect
 * degradation before a full outage, not just react to one.
 */
class GnssSampler {
    // TODO: FusedLocationProviderClient.requestLocationUpdates + GnssStatus.Callback
}
