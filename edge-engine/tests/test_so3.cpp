// Sign and branch-cut conventions for the attitude manifold. Every rotation
// bug in a strapdown filter eventually traces back to one of these six
// properties, so they are pinned before any dynamics are written.

#include "driftless/so3.h"

#include <gtest/gtest.h>

#include <random>

namespace {

using driftless::Quat;
using driftless::Vec3;
using driftless::so3::boxMinus;
using driftless::so3::boxPlus;
using driftless::so3::expMap;
using driftless::so3::logMap;

constexpr double kPi = 3.14159265358979323846;

TEST(So3, ExpLogRoundTripAcrossMagnitudes) {
    std::mt19937 rng(20260906);
    std::normal_distribution<double> gauss(0.0, 1.0);
    // Spans the hot 200 Hz regime (~1e-3 rad/step) through to near-pi, where
    // the quaternion sign representative starts to matter.
    for (double magnitude : {1e-12, 1e-9, 1e-6, 1e-3, 0.1, 1.0, 3.0, kPi - 1e-6}) {
        for (int trial = 0; trial < 25; ++trial) {
            Vec3 axis(gauss(rng), gauss(rng), gauss(rng));
            if (axis.norm() < 1e-9) continue;
            const Vec3 phi = axis.normalized() * magnitude;
            const Vec3 recovered = logMap(expMap(phi));
            EXPECT_LT((recovered - phi).norm(), 1e-10 * std::max(1.0, magnitude))
                << "magnitude=" << magnitude;
        }
    }
}

TEST(So3, ExpMapProducesUnitQuaternions) {
    std::mt19937 rng(20260907);
    std::normal_distribution<double> gauss(0.0, 2.0);
    for (int trial = 0; trial < 500; ++trial) {
        const Vec3 phi(gauss(rng), gauss(rng), gauss(rng));
        EXPECT_NEAR(expMap(phi).norm(), 1.0, 1e-12);
    }
}

TEST(So3, SmallAngleBranchAgreesWithTrigonometricBranch) {
    // The Taylor branch must not be a discontinuity. Straddle the threshold and
    // check both sides land on the same rotation matrix.
    const Vec3 axis = Vec3(1.0, -2.0, 0.5).normalized();
    for (double t : {9e-9, 1.1e-8}) {
        const Eigen::Matrix3d r = expMap(axis * t).toRotationMatrix();
        // Analytic Rodrigues for the same rotation.
        Eigen::Matrix3d skew;
        skew << 0, -axis.z(), axis.y(), axis.z(), 0, -axis.x(), -axis.y(),
            axis.x(), 0;
        const Eigen::Matrix3d expected = Eigen::Matrix3d::Identity() +
                                         std::sin(t) * skew +
                                         (1 - std::cos(t)) * skew * skew;
        EXPECT_LT((r - expected).cwiseAbs().maxCoeff(), 1e-14) << "t=" << t;
    }
}

TEST(So3, RotationMatrixIsOrthonormalWithUnitDeterminant) {
    // Guards against a proper rotation silently becoming a reflection, which
    // would flip a sign in the mechanization without any test failing.
    std::mt19937 rng(20260908);
    std::normal_distribution<double> gauss(0.0, 1.5);
    for (int trial = 0; trial < 200; ++trial) {
        const Eigen::Matrix3d r =
            expMap(Vec3(gauss(rng), gauss(rng), gauss(rng))).toRotationMatrix();
        EXPECT_LT((r * r.transpose() - Eigen::Matrix3d::Identity())
                      .cwiseAbs()
                      .maxCoeff(),
                  1e-12);
        EXPECT_NEAR(r.determinant(), 1.0, 1e-12);
    }
}

TEST(So3, BoxPlusAndBoxMinusAreInverses) {
    // The property the UKF actually depends on: compose an error onto the
    // nominal, decompose it back, get the same error.
    std::mt19937 rng(20260909);
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (int trial = 0; trial < 300; ++trial) {
        const Quat q = expMap(Vec3(gauss(rng), gauss(rng), gauss(rng)));
        const Vec3 delta = Vec3(gauss(rng), gauss(rng), gauss(rng)) * 0.01;
        EXPECT_LT((boxMinus(boxPlus(q, delta), q) - delta).norm(), 1e-12)
            << "trial=" << trial;
    }
}

TEST(So3, BoxPlusIsALocalPerturbation) {
    // delta is expressed in the BODY frame, not the nav frame. If this ever
    // flips to a left-multiply the filter still runs and quietly mis-rotates
    // every attitude correction, so it gets its own test.
    const Quat q = expMap(Vec3(0.0, 0.0, kPi / 2));  // 90 deg yaw
    const Vec3 delta(0.1, 0.0, 0.0);                 // roll, body frame
    EXPECT_LT((boxPlus(q, delta).coeffs() - (q * expMap(delta)).coeffs()).norm(),
              1e-15);
    // ... and is NOT the nav-frame equivalent.
    EXPECT_GT((boxPlus(q, delta).coeffs() - (expMap(delta) * q).coeffs()).norm(),
              1e-3);
}

TEST(So3, LogMapReturnsShortWayRoundForNegatedQuaternion) {
    const Vec3 phi = Vec3(0.3, -0.7, 1.1);
    const Quat q = expMap(phi);
    Quat negated = q;
    negated.coeffs() = -negated.coeffs();
    EXPECT_LT((logMap(negated) - phi).norm(), 1e-12);
}

}  // namespace
