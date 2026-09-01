"""Small geodesy helpers. Local-tangent-plane only -- no projection library."""

from __future__ import annotations

import numpy as np

R_EARTH_M = 6_378_137.0


def latlon_to_enu(lat_deg, lon_deg, lat0_deg=None, lon0_deg=None):
    """Equirectangular projection to local East/North metres about an origin.

    Accurate to well under a metre over the few-km spans of a single IO-VNBD
    sequence, which is far below the ~3-5 m GNSS noise we are labelling against.
    """
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    if lat0_deg is None:
        lat0_deg = float(np.nanmean(lat))
    if lon0_deg is None:
        lon0_deg = float(np.nanmean(lon))

    lat0_rad = np.deg2rad(lat0_deg)
    east = np.deg2rad(lon - lon0_deg) * R_EARTH_M * np.cos(lat0_rad)
    north = np.deg2rad(lat - lat0_deg) * R_EARTH_M
    return east, north, lat0_deg, lon0_deg


def wrap_pi(a):
    """Wrap angles in radians to (-pi, pi]."""
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def wrap_deg180(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0
