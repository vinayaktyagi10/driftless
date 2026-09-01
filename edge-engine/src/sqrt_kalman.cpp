#include "driftless/sqrt_kalman.h"

#include "driftless/cholesky_update.h"

#include <cmath>

namespace driftless::linalg {
namespace {

// Chi-squared upper-tail critical values. Only the degrees of freedom this
// engine measures in are tabulated: 1 (map-match cross-track), 2 (NHC lateral
// and vertical), 3 (GNSS position), 6 (GNSS position and velocity).
struct ChiSquaredEntry {
    int degrees_of_freedom;
    double at_95;
    double at_99;
    double at_999;
};

constexpr ChiSquaredEntry kChiSquaredTable[] = {
    {1, 3.841, 6.635, 10.828},
    {2, 5.991, 9.210, 13.816},
    {3, 7.815, 11.345, 16.266},
    {6, 12.592, 16.812, 22.458},
};

}  // namespace

SqrtKalmanUpdate arrayFormUpdate(
    const Eigen::Ref<const Eigen::MatrixXd>& sqrt_covariance,
    const Eigen::Ref<const Eigen::MatrixXd>& H,
    const Eigen::Ref<const Eigen::MatrixXd>& sqrt_R) {
    const Eigen::Index n = sqrt_covariance.rows();
    const Eigen::Index m = H.rows();

    // Pre-array. Triangularizing this is the whole algorithm; the blocks below
    // are then just read off.
    Eigen::MatrixXd pre_array = Eigen::MatrixXd::Zero(n + m, n + m);
    pre_array.topLeftCorner(m, m) = sqrt_R;
    pre_array.topRightCorner(m, n) = H * sqrt_covariance;
    pre_array.bottomRightCorner(n, n) = sqrt_covariance;

    // qrToLowerTriangular(A) returns L with L*L^T = A^T*A, so transposing gives
    // the L with L*L^T = pre_array * pre_array^T, which is what the identities
    // in the header are stated over.
    const Eigen::MatrixXd post_array =
        qrToLowerTriangular(pre_array.transpose());

    SqrtKalmanUpdate result;
    result.sqrt_innovation_covariance = post_array.topLeftCorner(m, m);
    result.sqrt_covariance_posterior = post_array.bottomRightCorner(n, n);

    // K_bar = P*H^T*S_nu^-T sits in the lower-left block, so the gain is one
    // triangular solve away: K = K_bar * S_nu^-1.
    const Eigen::MatrixXd gain_bar = post_array.bottomLeftCorner(n, m);
    result.gain = result.sqrt_innovation_covariance.transpose()
                      .triangularView<Eigen::Upper>()
                      .solve(gain_bar.transpose())
                      .transpose();
    return result;
}

double normalizedInnovationSquared(
    const Eigen::Ref<const Eigen::MatrixXd>& sqrt_innovation_covariance,
    const Eigen::Ref<const Eigen::VectorXd>& innovation) {
    // nu^T (S S^T)^-1 nu == ||S^-1 nu||^2, by forward substitution.
    const Eigen::VectorXd whitened =
        sqrt_innovation_covariance.triangularView<Eigen::Lower>().solve(innovation);
    return whitened.squaredNorm();
}

double chiSquaredThreshold(int degrees_of_freedom, double confidence) {
    for (const auto& entry : kChiSquaredTable) {
        if (entry.degrees_of_freedom != degrees_of_freedom) continue;
        if (std::abs(confidence - 0.95) < 1e-6) return entry.at_95;
        if (std::abs(confidence - 0.99) < 1e-6) return entry.at_99;
        if (std::abs(confidence - 0.999) < 1e-6) return entry.at_999;
        return -1.0;
    }
    return -1.0;
}

}  // namespace driftless::linalg
