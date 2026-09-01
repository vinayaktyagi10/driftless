#pragma once

#include "driftless/types.h"

namespace driftless::so3 {

// Exponential map: a rotation vector (axis * angle, radians) to the unit
// quaternion representing that rotation. Uses a Taylor expansion near zero,
// where sin(theta/2)/theta is 0/0.
[[nodiscard]] Quat expMap(const Vec3& rotation_vector);

// Logarithmic map: the inverse of expMap, returning the rotation vector with
// the smaller of the two equivalent magnitudes (|phi| <= pi).
[[nodiscard]] Vec3 logMap(const Quat& q);

// Manifold "addition": apply a local (body-frame) rotation-vector perturbation
// to an orientation. This is how an error-state delta gets injected into the
// nominal attitude.
[[nodiscard]] Quat boxPlus(const Quat& q, const Vec3& delta);

// Manifold "subtraction": the local rotation vector taking `reference` to `q`,
// i.e. the delta d such that boxPlus(reference, d) == q. This is how a
// propagated sigma point gets decomposed back into error coordinates.
[[nodiscard]] Vec3 boxMinus(const Quat& q, const Quat& reference);

}  // namespace driftless::so3
