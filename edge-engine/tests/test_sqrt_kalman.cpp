// The square-root update, checked against the textbook Kalman equations. The
// array form is compact to the point of being opaque -- three meaningful blocks
// fall out of one triangularization with no visible algebra -- so it is worth
// pinning every one of them against the form you would write on a whiteboard.

#include "driftless/sqrt_kalman.h"

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <gtest/gtest.h>

#include <random>

namespace {

using driftless::linalg::arrayFormUpdate;
using driftless::linalg::chiSquaredThreshold;
using driftless::linalg::normalizedInnovationSquared;

Eigen::MatrixXd randomSpd(int n, std::mt19937& rng) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    Eigen::MatrixXd m(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) m(i, j) = gauss(rng);
    return m * m.transpose() + n * Eigen::MatrixXd::Identity(n, n);
}

TEST(SqrtKalman, ReproducesTextbookKalmanEquations) {
    std::mt19937 rng(20260920);
    std::normal_distribution<double> gauss(0.0, 1.0);

    for (int n : {2, 5, 15}) {
        for (int m : {1, 2, 3, 6}) {
            if (m > n) continue;
            const Eigen::MatrixXd P = randomSpd(n, rng);
            const Eigen::MatrixXd R = randomSpd(m, rng);
            Eigen::MatrixXd H(m, n);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) H(i, j) = gauss(rng);

            const Eigen::MatrixXd S = Eigen::LLT<Eigen::MatrixXd>(P).matrixL();
            const Eigen::MatrixXd sqrt_R = Eigen::LLT<Eigen::MatrixXd>(R).matrixL();
            const auto result = arrayFormUpdate(S, H, sqrt_R);

            // Innovation covariance.
            const Eigen::MatrixXd expected_innovation = H * P * H.transpose() + R;
            const Eigen::MatrixXd actual_innovation =
                result.sqrt_innovation_covariance *
                result.sqrt_innovation_covariance.transpose();
            EXPECT_LT((actual_innovation - expected_innovation).cwiseAbs().maxCoeff(),
                      1e-9 * expected_innovation.cwiseAbs().maxCoeff())
                << "n=" << n << " m=" << m;

            // Kalman gain.
            const Eigen::MatrixXd expected_gain =
                P * H.transpose() * expected_innovation.inverse();
            EXPECT_LT((result.gain - expected_gain).cwiseAbs().maxCoeff(),
                      1e-9 * std::max(1.0, expected_gain.cwiseAbs().maxCoeff()))
                << "n=" << n << " m=" << m;

            // Posterior covariance, against the Joseph-equivalent form.
            const Eigen::MatrixXd expected_posterior =
                P - expected_gain * expected_innovation * expected_gain.transpose();
            const Eigen::MatrixXd actual_posterior =
                result.sqrt_covariance_posterior *
                result.sqrt_covariance_posterior.transpose();
            EXPECT_LT((actual_posterior - expected_posterior).cwiseAbs().maxCoeff(),
                      1e-9 * P.cwiseAbs().maxCoeff())
                << "n=" << n << " m=" << m;
        }
    }
}

TEST(SqrtKalman, PosteriorFactorIsLowerTriangularAndPositiveDefinite) {
    // The structural property the square-root form exists to guarantee. The
    // textbook P - K*H*P can go indefinite under a very informative
    // measurement; this must not.
    std::mt19937 rng(20260921);
    const int n = 15;
    const int m = 3;
    const Eigen::MatrixXd S =
        Eigen::LLT<Eigen::MatrixXd>(randomSpd(n, rng)).matrixL();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m, n);
    H.leftCols(m) = Eigen::MatrixXd::Identity(m, m);

    // An extremely precise measurement -- the regime that breaks naive forms.
    const Eigen::MatrixXd sqrt_R = 1e-8 * Eigen::MatrixXd::Identity(m, m);
    const auto result = arrayFormUpdate(S, H, sqrt_R);

    for (int i = 0; i < n; ++i) {
        EXPECT_GT(result.sqrt_covariance_posterior(i, i), 0.0) << "diagonal " << i;
        for (int j = i + 1; j < n; ++j)
            EXPECT_DOUBLE_EQ(result.sqrt_covariance_posterior(i, j), 0.0);
    }
}

TEST(SqrtKalman, UpdateReducesUncertaintyInTheMeasuredDirections) {
    // Sanity with a direction: measuring the first three states must shrink
    // their variance and must not inflate anything.
    std::mt19937 rng(20260922);
    const int n = 15;
    const int m = 3;
    const Eigen::MatrixXd P = randomSpd(n, rng);
    const Eigen::MatrixXd S = Eigen::LLT<Eigen::MatrixXd>(P).matrixL();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m, n);
    H.leftCols(m) = Eigen::MatrixXd::Identity(m, m);

    const auto result = arrayFormUpdate(S, H, Eigen::MatrixXd::Identity(m, m));
    const Eigen::MatrixXd posterior = result.sqrt_covariance_posterior *
                                      result.sqrt_covariance_posterior.transpose();

    for (int i = 0; i < m; ++i) EXPECT_LT(posterior(i, i), P(i, i)) << "state " << i;
    for (int i = 0; i < n; ++i) EXPECT_LE(posterior(i, i), P(i, i) + 1e-9);
}

TEST(SqrtKalman, NormalizedInnovationSquaredMatchesQuadraticForm) {
    std::mt19937 rng(20260923);
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (int m : {1, 2, 3, 6}) {
        const Eigen::MatrixXd cov = randomSpd(m, rng);
        const Eigen::MatrixXd sqrt_cov = Eigen::LLT<Eigen::MatrixXd>(cov).matrixL();
        Eigen::VectorXd innovation(m);
        for (int i = 0; i < m; ++i) innovation(i) = gauss(rng);

        const double expected =
            innovation.transpose() * cov.inverse() * innovation;
        EXPECT_NEAR(normalizedInnovationSquared(sqrt_cov, innovation), expected,
                    1e-9 * std::max(1.0, expected))
            << "m=" << m;
    }
}

TEST(SqrtKalman, ChiSquaredThresholdsMatchPublishedTable) {
    // Spot values from standard tables. If these drift, every outlier gate in
    // the engine silently changes its rejection rate.
    EXPECT_NEAR(chiSquaredThreshold(1, 0.99), 6.635, 1e-3);
    EXPECT_NEAR(chiSquaredThreshold(2, 0.99), 9.210, 1e-3);
    EXPECT_NEAR(chiSquaredThreshold(3, 0.99), 11.345, 1e-3);
    EXPECT_NEAR(chiSquaredThreshold(6, 0.99), 16.812, 1e-3);
    EXPECT_NEAR(chiSquaredThreshold(3, 0.95), 7.815, 1e-3);
    EXPECT_NEAR(chiSquaredThreshold(3, 0.999), 16.266, 1e-3);
    EXPECT_LT(chiSquaredThreshold(4, 0.99), 0.0) << "unsupported dof must signal";
}

}  // namespace
