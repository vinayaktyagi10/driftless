#include "driftless/so3.h"

#include <cmath>

namespace driftless::so3 {
namespace {
// Below this rotation magnitude the trigonometric forms lose precision to
// cancellation, so the Taylor expansions are used instead. At 200 Hz a
// typical per-step rotation is ~1e-3 rad, so this branch is the hot one.
constexpr double kSmallAngle = 1e-8;
}  // namespace

Quat expMap(const Vec3& rotation_vector) {
    const double theta = rotation_vector.norm();
    if (theta < kSmallAngle) {
        // cos(t/2) ~ 1 - t^2/8, and sin(t/2)/t ~ 1/2 - t^2/48.
        const Vec3 vec = rotation_vector * (0.5 - theta * theta / 48.0);
        return Quat(1.0 - theta * theta / 8.0, vec.x(), vec.y(), vec.z())
            .normalized();
    }
    const double half = 0.5 * theta;
    const Vec3 vec = rotation_vector * (std::sin(half) / theta);
    return Quat(std::cos(half), vec.x(), vec.y(), vec.z());
}

Vec3 logMap(const Quat& q) {
    // q and -q are the same rotation; picking the w >= 0 representative is what
    // guarantees the returned vector is the short way round.
    Quat n = q.normalized();
    if (n.w() < 0.0) n.coeffs() = -n.coeffs();

    const Vec3 vec = n.vec();
    const double sin_half = vec.norm();
    if (sin_half < kSmallAngle) {
        // theta ~ 2*sin(theta/2)/w to leading order; w ~ 1 here.
        return 2.0 * vec / n.w();
    }
    const double theta = 2.0 * std::atan2(sin_half, n.w());
    return vec * (theta / sin_half);
}

Quat boxPlus(const Quat& q, const Vec3& delta) {
    return (q * expMap(delta)).normalized();
}

Vec3 boxMinus(const Quat& q, const Quat& reference) {
    return logMap(reference.conjugate() * q);
}

}  // namespace driftless::so3
