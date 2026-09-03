"""Mount-rotation augmentation, and why the derived channels survive it.

The problem this addresses
-------------------------
Every phone in IO-VNBD lay flat, screen up: the mean accelerometer direction is
(0, 0, ~1) in all 51 runs. So the model has never seen a tilted handset, while
the product puts one in a dashboard mount at an arbitrary angle. Nine of the
fourteen input channels are raw body-frame axes and go out of distribution the
moment the phone is tilted.

The fix, and the reason it is exact
-----------------------------------
A mount is a FIXED rotation R between the handset frame and the vehicle. Gravity
is tracked by a linear filter (preprocess._causal_gravity), and a fixed rotation
commutes with a linear filter, so lowpass(R a) = R lowpass(a) exactly. Therefore
for the gravity direction g_hat:

    |R a|                      = |a|                    -> acc_norm   invariant
    (R a) . (R g_hat)          = a . g_hat              -> acc_vert   invariant
    sqrt(|Ra|^2 - (Ra.Rg)^2)   = same                   -> acc_horiz  invariant
    (R w) . (R g_hat)          = w . g_hat              -> gyro_vert  invariant
                                                        -> gyro_horiz invariant

The five derived channels are therefore *provably* mount-invariant, not
approximately so -- rotation of a whole run leaves them bit-comparable. That is
what `tests/test_augment.py` asserts on real data.

So augmentation only has to rotate the nine raw channels and leave the five
derived ones alone, which makes it cheap enough to do inside __getitem__.

AXIS ORDER WARNING
------------------
The feature vector stores the gyro triple in IO-VNBD's *label* order
(Yaw, Pitch, Roll), but the physical (X, Y, Z) order is a permutation of that --
`preprocess.GYRO_XYZ_COLUMNS` -- because the column headed "Pitch" is the
vertical axis. Rotating the stored slots directly would scramble the axes. The
permutation is derived from the channel names here so it cannot drift.
"""

from __future__ import annotations

import numpy as np

from .preprocess import DERIVED_CHANNELS, FEATURE_CHANNELS, GYRO_XYZ_COLUMNS

_F = list(FEATURE_CHANNELS)

# Indices of each physical 3-vector within the feature axis, in true X, Y, Z order.
ACC_XYZ_IDX = np.array([_F.index(c) for c in ("acc_x", "acc_y", "acc_z")])
GYRO_XYZ_IDX = np.array([_F.index(c) for c in GYRO_XYZ_COLUMNS])
GRAV_XYZ_IDX = np.array([_F.index(c) for c in ("grav_x", "grav_y", "grav_z")])
VECTOR_BLOCKS = (ACC_XYZ_IDX, GYRO_XYZ_IDX, GRAV_XYZ_IDX)

DERIVED_IDX = np.array([_F.index(c) for c in DERIVED_CHANNELS])
INVARIANT_CHANNELS = DERIVED_CHANNELS


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation on SO(3), via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))            # fix the QR sign ambiguity
    if np.linalg.det(q) < 0:            # reflection -> rotation
        q[:, 0] *= -1.0
    return q


def tilt_rotation(rng: np.random.Generator, max_tilt_deg: float = 60.0) -> np.ndarray:
    """A plausible dashboard mount: arbitrary heading, bounded tilt from level.

    Uniform-on-SO(3) includes upside-down phones, which no mount produces. This
    samples a random azimuth plus a tilt drawn up to `max_tilt_deg`, which is the
    realistic family and a harder test to pass honestly than a trivial one.
    """
    azim = rng.uniform(0.0, 2.0 * np.pi)
    tilt = np.deg2rad(rng.uniform(0.0, max_tilt_deg))
    spin = rng.uniform(0.0, 2.0 * np.pi)

    def rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def rx(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    return rz(azim) @ rx(tilt) @ rz(spin)


def rotate_window(x: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply a mount rotation to a (C, W) feature window.

    Rotates the raw acc/gyro/gravity triples; leaves the derived channels alone
    because they are invariant by construction (see the module docstring).
    """
    out = x.copy()
    for idx in VECTOR_BLOCKS:
        out[idx, :] = R @ x[idx, :]
    return out


def rotate_batch(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Same, for a (N, C, W) batch under one shared rotation."""
    out = X.copy()
    for idx in VECTOR_BLOCKS:
        out[:, idx, :] = np.einsum("ij,njw->niw", R, X[:, idx, :])
    return out


def invariant_channel_indices() -> np.ndarray:
    """Feature indices of the mount-invariant subset."""
    return DERIVED_IDX.copy()
