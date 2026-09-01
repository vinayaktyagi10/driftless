// The load-bearing numerical unit of the square-root UKF. A silent bug here
// corrupts every downstream estimate without ever throwing, so it is pinned
// against a naive reference that recomputes the factorization from scratch.

#include "driftless/cholesky_update.h"

#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include <random>

namespace {

using driftless::linalg::cholUpdate;
using driftless::linalg::qrToLowerTriangular;

// A random symmetric positive-definite matrix, and its lower Cholesky factor.
Eigen::MatrixXd randomSpd(int n, std::mt19937& rng) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    Eigen::MatrixXd m(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) m(i, j) = gauss(rng);
    // M*M^T is PSD; the diagonal boost makes it comfortably PD.
    return m * m.transpose() + n * Eigen::MatrixXd::Identity(n, n);
}

Eigen::VectorXd randomVec(int n, std::mt19937& rng) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v(i) = gauss(rng);
    return v;
}

// The thing cholUpdate must agree with: form the updated matrix explicitly and
// factor it directly. Deliberately the slow, obvious implementation.
Eigen::MatrixXd naiveUpdatedFactor(const Eigen::MatrixXd& A,
                                   const Eigen::VectorXd& u, double sigma) {
    Eigen::MatrixXd updated = A + sigma * u * u.transpose();
    return Eigen::LLT<Eigen::MatrixXd>(updated).matrixL();
}

TEST(CholUpdate, MatchesNaiveReferenceOnUpdate) {
    std::mt19937 rng(20260901);
    for (int n : {1, 2, 3, 7, 15}) {
        for (int trial = 0; trial < 20; ++trial) {
            const Eigen::MatrixXd A = randomSpd(n, rng);
            const Eigen::VectorXd u = randomVec(n, rng);
            Eigen::MatrixXd S = Eigen::LLT<Eigen::MatrixXd>(A).matrixL();

            ASSERT_TRUE(cholUpdate(S, u, 1.0)) << "n=" << n;
            const Eigen::MatrixXd expected = naiveUpdatedFactor(A, u, 1.0);
            EXPECT_LT((S - expected).cwiseAbs().maxCoeff(), 1e-9)
                << "n=" << n << " trial=" << trial;
        }
    }
}

TEST(CholUpdate, MatchesNaiveReferenceOnDowndate) {
    std::mt19937 rng(20260902);
    for (int n : {1, 2, 3, 7, 15}) {
        for (int trial = 0; trial < 20; ++trial) {
            const Eigen::MatrixXd A = randomSpd(n, rng);
            // Scale u down so A - u*u^T stays positive definite; a downdate
            // large enough to break that is covered separately below.
            const Eigen::VectorXd u = randomVec(n, rng) * 0.1;
            Eigen::MatrixXd S = Eigen::LLT<Eigen::MatrixXd>(A).matrixL();

            ASSERT_TRUE(cholUpdate(S, u, -1.0)) << "n=" << n;
            const Eigen::MatrixXd expected = naiveUpdatedFactor(A, u, -1.0);
            EXPECT_LT((S - expected).cwiseAbs().maxCoeff(), 1e-9)
                << "n=" << n << " trial=" << trial;
        }
    }
}

// The UKF's centre-point weight is not +/-1 -- with alpha=1e-3 it is on the
// order of -1e6 -- so arbitrary sigma magnitudes have to be exact, not just
// the unit-weight case.
TEST(CholUpdate, HandlesArbitrarySigmaMagnitude) {
    std::mt19937 rng(20260903);
    for (double sigma : {0.001, 0.5, 3.7, 1000.0, -0.001, -0.05}) {
        const int n = 15;
        const Eigen::MatrixXd A = randomSpd(n, rng);
        const Eigen::VectorXd u = randomVec(n, rng) * 0.05;
        Eigen::MatrixXd S = Eigen::LLT<Eigen::MatrixXd>(A).matrixL();

        ASSERT_TRUE(cholUpdate(S, u, sigma)) << "sigma=" << sigma;
        const Eigen::MatrixXd expected = naiveUpdatedFactor(A, u, sigma);
        EXPECT_LT((S - expected).cwiseAbs().maxCoeff(), 1e-8) << "sigma=" << sigma;
    }
}

TEST(CholUpdate, ResultStaysLowerTriangularWithPositiveDiagonal) {
    std::mt19937 rng(20260904);
    const int n = 15;
    Eigen::MatrixXd S = Eigen::LLT<Eigen::MatrixXd>(randomSpd(n, rng)).matrixL();
    for (int step = 0; step < 200; ++step) {
        ASSERT_TRUE(cholUpdate(S, randomVec(n, rng) * 0.01, 1.0));
    }
    for (int i = 0; i < n; ++i) {
        EXPECT_GT(S(i, i), 0.0) << "diagonal " << i;
        for (int j = i + 1; j < n; ++j) EXPECT_DOUBLE_EQ(S(i, j), 0.0);
    }
}

// A downdate that removes more than the covariance contains must be reported,
// not silently turned into NaNs -- the filter needs the chance to react.
TEST(CholUpdate, ReportsFailureWhenDowndateBreaksPositiveDefiniteness) {
    const int n = 4;
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd u = Eigen::VectorXd::Zero(n);
    u(0) = 10.0;  // removes 100 from a unit variance
    EXPECT_FALSE(cholUpdate(S, u, -1.0));
}

TEST(QrToLowerTriangular, ReproducesFactorOfNormalEquations) {
    std::mt19937 rng(20260905);
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (int n : {2, 5, 15}) {
        const int rows = 3 * n;
        Eigen::MatrixXd A(rows, n);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < n; ++j) A(i, j) = gauss(rng);

        const Eigen::MatrixXd S = qrToLowerTriangular(A);
        const Eigen::MatrixXd expected =
            Eigen::LLT<Eigen::MatrixXd>(A.transpose() * A).matrixL();

        EXPECT_LT((S - expected).cwiseAbs().maxCoeff(), 1e-9) << "n=" << n;
        for (int i = 0; i < n; ++i) {
            EXPECT_GT(S(i, i), 0.0);
            for (int j = i + 1; j < n; ++j) EXPECT_DOUBLE_EQ(S(i, j), 0.0);
        }
    }
}

}  // namespace
