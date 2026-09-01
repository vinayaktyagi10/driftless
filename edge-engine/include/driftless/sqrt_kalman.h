#pragma once

#include <Eigen/Core>

namespace driftless::linalg {

// Result of one square-root measurement update, all three blocks read straight
// out of a single triangularization -- see arrayFormUpdate.
struct SqrtKalmanUpdate {
    // Lower-triangular Cholesky factor of the innovation covariance
    // (H*P*H^T + R). Needed for the NIS gate, which is why it is returned
    // rather than consumed internally: the caller has to be able to reject a
    // measurement *before* applying it.
    Eigen::MatrixXd sqrt_innovation_covariance;
    // Kalman gain, n x m.
    Eigen::MatrixXd gain;
    // Lower-triangular factor of the posterior covariance. Positive definite by
    // construction -- there is no subtraction anywhere in the derivation, which
    // is the entire reason for preferring this form over P - K*H*P.
    Eigen::MatrixXd sqrt_covariance_posterior;
};

// Kailath array form of the linear Kalman update, in square-root covariance
// coordinates. Given the prior factor S (P = S*S^T), a linear measurement
// model H, and the measurement noise factor sqrt_R (R = sqrt_R*sqrt_R^T), it
// triangularizes
//
//     A = [ sqrt_R   H*S ]        ->        [ S_nu   0  ]
//         [   0       S  ]                  [ K_bar  S+ ]
//
// and reads the three blocks off the result. The identities that fall out:
//
//     S_nu*S_nu^T = H*P*H^T + R          (innovation covariance)
//     K           = K_bar * S_nu^-1      (Kalman gain)
//     S+*S+^T     = P - K*(H*P*H^T+R)*K^T
//
// Nothing is ever inverted and nothing is ever subtracted, so the posterior
// cannot lose positive-definiteness the way the textbook P - K*H*P form can.
[[nodiscard]] SqrtKalmanUpdate arrayFormUpdate(
    const Eigen::Ref<const Eigen::MatrixXd>& sqrt_covariance,
    const Eigen::Ref<const Eigen::MatrixXd>& H,
    const Eigen::Ref<const Eigen::MatrixXd>& sqrt_R);

// Normalized innovation squared: nu^T * (S_nu*S_nu^T)^-1 * nu, computed by
// triangular solve rather than by forming the inverse. Under the hypothesis
// that the filter and the measurement are mutually consistent this is
// chi-squared distributed with m degrees of freedom, which is what makes it a
// usable outlier test.
[[nodiscard]] double normalizedInnovationSquared(
    const Eigen::Ref<const Eigen::MatrixXd>& sqrt_innovation_covariance,
    const Eigen::Ref<const Eigen::VectorXd>& innovation);

// Upper-tail chi-squared critical values, tabulated for the handful of degrees
// of freedom this engine actually uses. A table rather than an incomplete-gamma
// implementation because the alternative is 200 lines of special-function code
// to serve six lookups, and a wrong entry here is far easier to spot.
// Returns a negative value for an unsupported (dof, confidence) pair.
[[nodiscard]] double chiSquaredThreshold(int degrees_of_freedom,
                                         double confidence);

}  // namespace driftless::linalg
