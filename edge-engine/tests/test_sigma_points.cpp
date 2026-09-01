// The unscented transform's moment-matching identities are exact, not
// approximate: the sigma set must reproduce the mean and covariance it was
// drawn from to machine precision. That makes this the cheapest possible check
// on the weights and the spread factor, entirely independent of any dynamics --
// so when a predict test fails later, this one having passed rules out a whole
// class of cause.

#include "driftless/ukf_fusion_engine.h"

#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include <random>

namespace {

using driftless::ImuNoiseParams;
using driftless::NavState;
using driftless::UkfFusionEngine;
using ErrorMatrix = UkfFusionEngine::ErrorMatrix;
using ErrorVector = UkfFusionEngine::ErrorVector;

ErrorMatrix randomSqrtCovariance(std::mt19937& rng) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    ErrorMatrix m;
    for (int i = 0; i < UkfFusionEngine::kStateDim; ++i)
        for (int j = 0; j < UkfFusionEngine::kStateDim; ++j) m(i, j) = gauss(rng);
    const ErrorMatrix spd = m * m.transpose() +
                            UkfFusionEngine::kStateDim *
                                ErrorMatrix::Identity();
    return Eigen::LLT<ErrorMatrix>(spd).matrixL();
}

UkfFusionEngine makeEngine(const ErrorMatrix& s,
                           const UkfFusionEngine::Config& cfg = {}) {
    return UkfFusionEngine(NavState{}, s, ImuNoiseParams::fogGrade(), cfg);
}

TEST(SigmaPoints, WeightsSumToOne) {
    // sum(Wm) == 1 exactly. sum(Wc) is NOT 1: the 2n outer weights already
    // sum to (1 - Wm0), and Wc0 adds (Wm0 + 1 - alpha^2 + beta) on top, giving
    // 2 + beta - alpha^2. That deliberate excess is what beta buys -- it
    // inflates the covariance to account for higher-order moments -- so a
    // covariance-weight sum of exactly 1 would mean beta had been dropped.
    for (double alpha : {1e-3, 0.1, 0.5, 1.0}) {
        UkfFusionEngine::Config cfg;
        cfg.alpha = alpha;
        const auto engine = makeEngine(ErrorMatrix::Identity(), cfg);

        const double sum_mean =
            engine.weightMean0() + 2 * UkfFusionEngine::kStateDim * engine.weightI();
        EXPECT_NEAR(sum_mean, 1.0, 1e-9) << "alpha=" << alpha;

        const double sum_cov =
            engine.weightCov0() + 2 * UkfFusionEngine::kStateDim * engine.weightI();
        EXPECT_NEAR(sum_cov, 2.0 + cfg.beta - alpha * alpha, 1e-9)
            << "alpha=" << alpha;
    }
}

TEST(SigmaPoints, CentreWeightIsNegativeForSmallAlpha) {
    // Not incidental: this is precisely why the covariance step needs a
    // Cholesky downdate. If this ever turns positive the downdate path in
    // predict() stops being exercised and the SR-UKF is only half tested.
    const auto engine = makeEngine(ErrorMatrix::Identity());
    EXPECT_LT(engine.weightCov0(), 0.0);
    EXPECT_LT(engine.weightMean0(), 0.0);
    EXPECT_GT(engine.weightI(), 0.0);
}

TEST(SigmaPoints, WeightedMeanRecoversZero) {
    std::mt19937 rng(20260910);
    for (int trial = 0; trial < 20; ++trial) {
        const ErrorMatrix s = randomSqrtCovariance(rng);
        const auto engine = makeEngine(s);
        const auto points = UkfFusionEngine::sigmaPoints(s, engine.gamma());

        ErrorVector mean = engine.weightMean0() * points[0];
        for (int i = 1; i < UkfFusionEngine::kSigmaPoints; ++i)
            mean += engine.weightI() * points[i];

        EXPECT_LT(mean.cwiseAbs().maxCoeff(), 1e-12) << "trial=" << trial;
    }
}

TEST(SigmaPoints, WeightedOuterProductRecoversCovariance) {
    // sum_i Wc_i * chi_i * chi_i^T == S*S^T. Catches a wrong gamma or a wrong
    // weight_i, either of which would otherwise show up only as a covariance
    // that inflates at a subtly wrong rate -- invisible without ground truth.
    std::mt19937 rng(20260911);
    for (int trial = 0; trial < 20; ++trial) {
        const ErrorMatrix s = randomSqrtCovariance(rng);
        const auto engine = makeEngine(s);
        const auto points = UkfFusionEngine::sigmaPoints(s, engine.gamma());

        ErrorMatrix reconstructed =
            engine.weightCov0() * points[0] * points[0].transpose();
        for (int i = 1; i < UkfFusionEngine::kSigmaPoints; ++i)
            reconstructed += engine.weightI() * points[i] * points[i].transpose();

        const ErrorMatrix expected = s * s.transpose();
        const double scale = expected.cwiseAbs().maxCoeff();
        EXPECT_LT((reconstructed - expected).cwiseAbs().maxCoeff(), 1e-9 * scale)
            << "trial=" << trial;
    }
}

TEST(SigmaPoints, SetIsSymmetricAboutTheCentre) {
    std::mt19937 rng(20260912);
    const ErrorMatrix s = randomSqrtCovariance(rng);
    const auto engine = makeEngine(s);
    const auto points = UkfFusionEngine::sigmaPoints(s, engine.gamma());

    EXPECT_EQ(points[0].cwiseAbs().maxCoeff(), 0.0);
    for (int i = 0; i < UkfFusionEngine::kStateDim; ++i) {
        EXPECT_LT((points[1 + i] + points[1 + UkfFusionEngine::kStateDim + i])
                      .cwiseAbs()
                      .maxCoeff(),
                  1e-12);
    }
}

TEST(ComposeDecompose, AreInversesAcrossTheFullState) {
    // predict() relies on decompose(compose(x, e)) == e for every block,
    // including the manifold one. If this breaks, the error mean silently
    // stops meaning what the covariance says it means.
    std::mt19937 rng(20260913);
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (int trial = 0; trial < 200; ++trial) {
        NavState nominal;
        nominal.position = driftless::Vec3(gauss(rng), gauss(rng), gauss(rng));
        nominal.velocity = driftless::Vec3(gauss(rng), gauss(rng), gauss(rng));
        nominal.orientation = driftless::so3::expMap(
            driftless::Vec3(gauss(rng), gauss(rng), gauss(rng)));
        nominal.accel_bias = driftless::Vec3(gauss(rng), gauss(rng), gauss(rng)) * 0.01;
        nominal.gyro_bias = driftless::Vec3(gauss(rng), gauss(rng), gauss(rng)) * 0.001;

        ErrorVector error;
        for (int i = 0; i < UkfFusionEngine::kStateDim; ++i) error(i) = gauss(rng) * 0.05;

        const ErrorVector recovered = UkfFusionEngine::decompose(
            UkfFusionEngine::compose(nominal, error), nominal);
        EXPECT_LT((recovered - error).cwiseAbs().maxCoeff(), 1e-12)
            << "trial=" << trial;
    }
}

}  // namespace
