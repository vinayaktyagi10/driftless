#include "driftless/imu_derived.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace driftless;

TEST(ImuDerivedTest, PythagoreanIdentityHolds) {
    Vec3 accel(1.2, -3.4, 9.8);
    Vec3 gyro(0.1, 0.02, -0.5);
    Vec3 gravity(0.3, -0.1, 9.8);

    auto d = computeImuDerived(accel, gyro, gravity);
    EXPECT_NEAR(d.acc_vert * d.acc_vert + d.acc_horiz * d.acc_horiz, accel.squaredNorm(), 1e-9);
    EXPECT_NEAR(d.gyro_vert * d.gyro_vert + d.gyro_horiz * d.gyro_horiz, gyro.squaredNorm(), 1e-9);
}

TEST(ImuDerivedTest, AllVerticalWhenAlignedWithGravity) {
    Vec3 gravity(0.0, 0.0, 1.0);
    Vec3 accel(0.0, 0.0, 9.8);
    Vec3 gyro(0.0, 0.0, 0.0);

    auto d = computeImuDerived(accel, gyro, gravity);
    EXPECT_NEAR(std::abs(d.acc_vert), accel.norm(), 1e-9);
    EXPECT_NEAR(d.acc_horiz, 0.0, 1e-9);
}

TEST(ImuDerivedTest, AllHorizontalWhenPerpendicularToGravity) {
    Vec3 gravity(0.0, 0.0, 1.0);
    Vec3 accel(5.0, 0.0, 0.0);
    Vec3 gyro(0.0, 0.0, 0.0);

    auto d = computeImuDerived(accel, gyro, gravity);
    EXPECT_NEAR(d.acc_vert, 0.0, 1e-9);
    EXPECT_NEAR(d.acc_horiz, accel.norm(), 1e-9);
}

TEST(GravityLowpassTest, FirstSampleSeedsStateExactly) {
    GravityLowpass lp(10.0);
    Vec3 first(0.1, 0.2, 9.8);
    Vec3 out = lp.push(first, 0.1);
    EXPECT_TRUE(out.isApprox(first));
}

TEST(GravityLowpassTest, ConvergesTowardConstantInput) {
    GravityLowpass lp(10.0);
    Vec3 target(0.0, 0.0, 9.80665);
    Vec3 out = lp.push(Vec3(5.0, 5.0, 5.0), 0.1);
    for (int i = 0; i < 2000; ++i) {  // 200s >> 3*tau
        out = lp.push(target, 0.1);
    }
    EXPECT_TRUE(out.isApprox(target, 1e-3));
}