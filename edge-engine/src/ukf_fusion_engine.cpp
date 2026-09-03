#include "driftless/ukf_fusion_engine.h"

#include "driftless/cholesky_update.h"
#include "driftless/so3.h"
#include "driftless/sqrt_kalman.h"

#include <array>
#include <cmath>

namespace driftless {
namespace {
constexpr double kNanosPerSecond = 1e9;
}  // namespace

UkfFusionEngine::UkfFusionEngine(const NavState& initial_state,
                                 const ErrorMatrix& initial_covariance_sqrt,
                                 const ImuNoiseParams& noise,
                                 const Config& config)
    : nominal_(initial_state),
      sqrt_covariance_(initial_covariance_sqrt),
      noise_(noise),
      config_(config) {
    nominal_.orientation.normalize();
    computeWeights();
}

UkfFusionEngine::UkfFusionEngine(const NavState& initial_state,
                                 const ErrorMatrix& initial_covariance_sqrt,
                                 const ImuNoiseParams& noise)
    : UkfFusionEngine(initial_state, initial_covariance_sqrt, noise, Config{}) {}

void UkfFusionEngine::computeWeights() {
    const double n = static_cast<double>(kStateDim);
    const double lambda =
        config_.alpha * config_.alpha * (n + config_.kappa) - n;
    const double n_plus_lambda = n + lambda;

    gamma_ = std::sqrt(n_plus_lambda);
    weight_mean_0_ = lambda / n_plus_lambda;
    weight_cov_0_ =
        weight_mean_0_ + 1.0 - config_.alpha * config_.alpha + config_.beta;
    weight_i_ = 0.5 / n_plus_lambda;
}

std::array<UkfFusionEngine::ErrorVector, UkfFusionEngine::kSigmaPoints>
UkfFusionEngine::sigmaPoints(const ErrorMatrix& sqrt_covariance, double gamma) {
    std::array<ErrorVector, kSigmaPoints> points;
    points[0] = ErrorVector::Zero();
    for (int i = 0; i < kStateDim; ++i) {
        const ErrorVector scaled_column = gamma * sqrt_covariance.col(i);
        points[1 + i] = scaled_column;
        points[1 + kStateDim + i] = -scaled_column;
    }
    return points;
}

NavState UkfFusionEngine::mechanize(const NavState& s, const Vec3& accel,
                                    const Vec3& gyro, double dt,
                                    const Vec3& gravity_ned) {
    // Bias-corrected measurements. The accelerometer reports specific force,
    // which is proper acceleration -- it does not see gravity, so gravity has
    // to be added back once the measurement is rotated into the nav frame.
    const Vec3 omega = gyro - s.gyro_bias;
    const Vec3 force = accel - s.accel_bias;
    const Vec3 accel_ned = s.orientation * force + gravity_ned;

    NavState out = s;
    // Attitude at the start of the interval is used for the whole interval --
    // first-order, which at 200Hz costs far less than the noise floor. Revisit
    // only if this ever runs at a materially lower rate.
    out.position = s.position + s.velocity * dt + 0.5 * accel_ned * dt * dt;
    out.velocity = s.velocity + accel_ned * dt;
    out.orientation = so3::boxPlus(s.orientation, omega * dt);
    // Biases are modelled as random walks: unchanged in the mean, driven only
    // by their process noise in Q.
    return out;
}

NavState UkfFusionEngine::compose(const NavState& nominal,
                                  const ErrorVector& error) {
    NavState out = nominal;
    out.position += error.segment<3>(kPositionIndex);
    out.velocity += error.segment<3>(kVelocityIndex);
    out.orientation =
        so3::boxPlus(nominal.orientation, error.segment<3>(kAttitudeIndex));
    out.accel_bias += error.segment<3>(kAccelBiasIndex);
    out.gyro_bias += error.segment<3>(kGyroBiasIndex);
    return out;
}

UkfFusionEngine::ErrorVector UkfFusionEngine::decompose(const NavState& s,
                                                        const NavState& nominal) {
    ErrorVector error;
    error.segment<3>(kPositionIndex) = s.position - nominal.position;
    error.segment<3>(kVelocityIndex) = s.velocity - nominal.velocity;
    error.segment<3>(kAttitudeIndex) =
        so3::boxMinus(s.orientation, nominal.orientation);
    error.segment<3>(kAccelBiasIndex) = s.accel_bias - nominal.accel_bias;
    error.segment<3>(kGyroBiasIndex) = s.gyro_bias - nominal.gyro_bias;
    return error;
}

bool UkfFusionEngine::predict(const ImuSample& sample) {
    if (!last_timestamp_nanos_.has_value()) {
        // Nothing to integrate over yet -- the first sample only establishes
        // the time origin.
        last_timestamp_nanos_ = sample.timestamp_nanos;
        return true;
    }

    const double dt =
        static_cast<double>(sample.timestamp_nanos - *last_timestamp_nanos_) /
        kNanosPerSecond;
    if (!(dt > 0.0) || dt > config_.max_step_seconds) return false;

    // --- 1. Sigma points, drawn in error space about zero -------------------
    // The mean error is zero by construction (it was injected and reset at the
    // end of the previous step), so the points are just +/- the scaled columns
    // of S with the centre at the origin.
    std::array<ErrorVector, kSigmaPoints> propagated_errors;
    std::array<NavState, kSigmaPoints> propagated_states;

    const std::array<ErrorVector, kSigmaPoints> chi =
        sigmaPoints(sqrt_covariance_, gamma_);

    for (int i = 0; i < kSigmaPoints; ++i) {
        // --- 2 & 3. Compose onto the nominal, then propagate through the
        // strapdown mechanization. Each sigma point is a full physical state
        // while it is being propagated; the nonlinearity is applied to the
        // state, not to the error.
        propagated_states[i] =
            mechanize(compose(nominal_, chi[i]), sample.accel, sample.gyro, dt,
                      config_.gravity_ned);
    }

    // --- 4. The propagated centre point IS the new nominal ------------------
    const NavState propagated_nominal = propagated_states[0];

    // --- 5. Decompose back to error coordinates about the NEW nominal -------
    // This is what makes step 6 legal: every entry below is now a tangent-space
    // vector, so it can be averaged arithmetically. Entry 0 is zero by
    // construction; it is computed rather than assumed so that a bug in
    // compose/decompose shows up as a test failure instead of cancelling out.
    for (int i = 0; i < kSigmaPoints; ++i) {
        propagated_errors[i] =
            decompose(propagated_states[i], propagated_nominal);
    }

    // --- 6. Weighted mean of the error ------------------------------------
    ErrorVector mean_error = weight_mean_0_ * propagated_errors[0];
    for (int i = 1; i < kSigmaPoints; ++i) {
        mean_error += weight_i_ * propagated_errors[i];
    }

    // --- 7. Square-root covariance update ----------------------------------
    // Stack the weighted deviations of the 2n outer points on top of sqrt(Q),
    // and take the QR. P is never formed, so its condition number is never
    // squared. weight_i_ is always positive, so this half needs no downdate.
    const double sqrt_dt = std::sqrt(dt);
    ErrorVector sqrt_q;
    // Position gets a small direct term. Physically its uncertainty grows
    // through velocity (the dt^1.5/sqrt(3) coefficient is the second-order
    // contribution of the same accelerometer noise), but it also keeps the
    // block from being exactly rank-deficient, which matters numerically.
    sqrt_q.segment<3>(kPositionIndex).setConstant(
        noise_.accel_noise_density * dt * sqrt_dt / std::sqrt(3.0));
    sqrt_q.segment<3>(kVelocityIndex).setConstant(noise_.accel_noise_density * sqrt_dt);
    sqrt_q.segment<3>(kAttitudeIndex).setConstant(noise_.gyro_noise_density * sqrt_dt);
    sqrt_q.segment<3>(kAccelBiasIndex).setConstant(noise_.accel_bias_random_walk * sqrt_dt);
    sqrt_q.segment<3>(kGyroBiasIndex).setConstant(noise_.gyro_bias_random_walk * sqrt_dt);

    Eigen::MatrixXd stacked(2 * kStateDim + kStateDim, kStateDim);
    const double sqrt_weight_i = std::sqrt(weight_i_);
    for (int i = 1; i < kSigmaPoints; ++i) {
        stacked.row(i - 1) =
            sqrt_weight_i * (propagated_errors[i] - mean_error).transpose();
    }
    stacked.bottomRows(kStateDim) = sqrt_q.asDiagonal();

    ErrorMatrix new_sqrt_covariance = linalg::qrToLowerTriangular(stacked);

    // The centre point carries weight W0c, which is large and negative for
    // alpha << 1 -- hence a downdate. If it fails the covariance has genuinely
    // left the positive-definite cone and the caller needs to know; silently
    // continuing would mean propagating a factor that no longer factors
    // anything.
    Eigen::MatrixXd factor = new_sqrt_covariance;
    const Eigen::VectorXd centre_deviation = propagated_errors[0] - mean_error;
    if (!linalg::cholUpdate(factor, centre_deviation, weight_cov_0_)) {
        return false;
    }

    // --- 8. Inject the mean error into the nominal and reset ---------------
    // The error mean is folded into the nominal state, so the error is zero
    // again for the next step while the covariance is retained.
    nominal_ = compose(propagated_nominal, mean_error);
    nominal_.orientation.normalize();
    sqrt_covariance_ = factor;
    last_timestamp_nanos_ = sample.timestamp_nanos;
    return true;
}

}  // namespace driftless

namespace driftless {

// --- Generic updates -------------------------------------------------------

UkfFusionEngine::UpdateOutcome UkfFusionEngine::updateLinear(
    const Eigen::MatrixXd& H, const Eigen::VectorXd& innovation,
    const Eigen::MatrixXd& sqrt_R, double gate_confidence) {
    const auto result = linalg::arrayFormUpdate(sqrt_covariance_, H, sqrt_R);

    const double nis = linalg::normalizedInnovationSquared(
        result.sqrt_innovation_covariance, innovation);
    diagnostics_.last_nis = nis;
    if (!std::isfinite(nis)) return UpdateOutcome::kNumericalFailure;

    const double threshold = linalg::chiSquaredThreshold(
        static_cast<int>(innovation.size()), gate_confidence);
    // A missing table entry must not silently disable the gate.
    if (threshold < 0.0) return UpdateOutcome::kNumericalFailure;
    if (nis > threshold) return UpdateOutcome::kRejectedByGate;

    // The prior error mean is zero, so the correction IS the posterior mean.
    const ErrorVector correction = result.gain * innovation;
    nominal_ = compose(nominal_, correction);
    nominal_.orientation.normalize();
    sqrt_covariance_ = result.sqrt_covariance_posterior;
    return UpdateOutcome::kApplied;
}

UkfFusionEngine::UpdateOutcome UkfFusionEngine::updateUnscented(
    const MeasurementFunction& h, const Eigen::VectorXd& z,
    const Eigen::MatrixXd& sqrt_R, double gate_confidence) {
    const Eigen::Index m = z.size();
    const auto chi = sigmaPoints(sqrt_covariance_, gamma_);

    std::array<Eigen::VectorXd, kSigmaPoints> predicted;
    for (int i = 0; i < kSigmaPoints; ++i) {
        predicted[i] = h(compose(nominal_, chi[i]));
    }

    Eigen::VectorXd predicted_mean = weight_mean_0_ * predicted[0];
    for (int i = 1; i < kSigmaPoints; ++i) {
        predicted_mean += weight_i_ * predicted[i];
    }

    // Innovation covariance factor, same array construction as the predict
    // step: stack the weighted deviations of the outer points on top of the
    // measurement noise factor, triangularize, then fold in the centre point
    // with a rank-1 update whose weight may be negative.
    Eigen::MatrixXd stacked(2 * kStateDim + m, m);
    const double sqrt_weight_i = std::sqrt(weight_i_);
    for (int i = 1; i < kSigmaPoints; ++i) {
        stacked.row(i - 1) =
            sqrt_weight_i * (predicted[i] - predicted_mean).transpose();
    }
    stacked.bottomRows(m) = sqrt_R.transpose();

    Eigen::MatrixXd sqrt_innovation = linalg::qrToLowerTriangular(stacked);
    const Eigen::VectorXd centre_deviation = predicted[0] - predicted_mean;
    if (!linalg::cholUpdate(sqrt_innovation, centre_deviation, weight_cov_0_)) {
        return UpdateOutcome::kNumericalFailure;
    }

    const Eigen::VectorXd innovation = z - predicted_mean;
    const double nis =
        linalg::normalizedInnovationSquared(sqrt_innovation, innovation);
    diagnostics_.last_nis = nis;
    if (!std::isfinite(nis)) return UpdateOutcome::kNumericalFailure;

    const double threshold =
        linalg::chiSquaredThreshold(static_cast<int>(m), gate_confidence);
    if (threshold < 0.0) return UpdateOutcome::kNumericalFailure;
    if (nis > threshold) return UpdateOutcome::kRejectedByGate;

    // Cross-covariance. The prior error mean is zero, so the state deviations
    // are the sigma points themselves; chi[0] is zero and drops out.
    Eigen::MatrixXd cross_covariance = Eigen::MatrixXd::Zero(kStateDim, m);
    for (int i = 1; i < kSigmaPoints; ++i) {
        cross_covariance +=
            weight_i_ * chi[i] * (predicted[i] - predicted_mean).transpose();
    }

    // K = P_xz * (S_zz S_zz^T)^-1, by two triangular solves rather than an
    // inverse.
    const Eigen::MatrixXd temp = sqrt_innovation.triangularView<Eigen::Lower>()
                                     .solve(cross_covariance.transpose());
    const Eigen::MatrixXd gain = sqrt_innovation.transpose()
                                     .triangularView<Eigen::Upper>()
                                     .solve(temp)
                                     .transpose();

    // Covariance downdate, one column of K*S_zz at a time. This is the step
    // that can legitimately fail on a very informative measurement, and the
    // filter must be left untouched if it does -- hence the scratch copy.
    Eigen::MatrixXd factor = sqrt_covariance_;
    const Eigen::MatrixXd downdate_columns = gain * sqrt_innovation;
    for (Eigen::Index j = 0; j < m; ++j) {
        const Eigen::VectorXd column = downdate_columns.col(j);
        if (!linalg::cholUpdate(factor, column, -1.0)) {
            return UpdateOutcome::kNumericalFailure;
        }
    }

    const ErrorVector correction = gain * innovation;
    nominal_ = compose(nominal_, correction);
    nominal_.orientation.normalize();
    sqrt_covariance_ = factor;
    return UpdateOutcome::kApplied;
}

// --- GNSS ------------------------------------------------------------------

Eigen::MatrixXd UkfFusionEngine::gnssSqrtNoise(const GnssFix& fix,
                                               bool use_velocity) const {
    const auto& p = config_.gnss;

    const double horizontal =
        std::max(fix.horizontal_accuracy_m, p.min_horizontal_accuracy_m);
    const double vertical =
        std::max(fix.vertical_accuracy_m, p.min_vertical_accuracy_m);

    // Inflate as satellite count falls below open-sky. Quadratic rather than
    // linear because the geometry degrades faster than the count suggests: the
    // satellites lost first in a canyon are the low-elevation ones that were
    // doing most of the work for horizontal accuracy.
    const int deficit = std::max(0, p.healthy_satellite_count - fix.satellites_used);
    const double inflation =
        1.0 + p.satellite_deficit_inflation * static_cast<double>(deficit) *
                  static_cast<double>(deficit);

    const Eigen::Index m = use_velocity ? 6 : 3;
    Eigen::VectorXd sigma(m);
    sigma(0) = horizontal * inflation;
    sigma(1) = horizontal * inflation;
    sigma(2) = vertical * inflation;
    if (use_velocity) {
        const double speed =
            std::max(fix.speed_accuracy_mps, p.min_speed_accuracy_mps) * inflation;
        sigma.segment<3>(3).setConstant(speed);
    }
    return Eigen::MatrixXd(sigma.asDiagonal());
}

UkfFusionEngine::UpdateOutcome UkfFusionEngine::updateGnss(const GnssFix& fix) {
    const bool use_velocity = fix.has_velocity;
    const Eigen::Index m = use_velocity ? 6 : 3;

    // h(dx) = dp (and dv), so H is a pair of identity blocks. This is what
    // makes the linear update exact rather than merely adequate.
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m, kStateDim);
    H.block<3, 3>(0, kPositionIndex) = Eigen::Matrix3d::Identity();
    if (use_velocity) {
        H.block<3, 3>(3, kVelocityIndex) = Eigen::Matrix3d::Identity();
    }

    Eigen::VectorXd innovation(m);
    innovation.segment<3>(0) = fix.position.vector() - nominal_.position;
    if (use_velocity) {
        innovation.segment<3>(3) = fix.velocity_ned - nominal_.velocity;
    }

    const UpdateOutcome outcome = updateLinear(
        H, innovation, gnssSqrtNoise(fix, use_velocity), config_.gnss.gate_confidence);
    if (outcome == UpdateOutcome::kApplied) {
        ++diagnostics_.gnss_applied;
    } else if (outcome == UpdateOutcome::kRejectedByGate) {
        ++diagnostics_.gnss_rejected;
    }
    return outcome;
}

// --- Map matching ----------------------------------------------------------

UkfFusionEngine::UpdateOutcome UkfFusionEngine::updateMapMatch(
    const MapMatchResult& match) {
    const auto& p = config_.map_match;
    if (!match.matched || match.distance_to_road_m > p.max_distance_to_road_m) {
        ++diagnostics_.map_match_skipped;
        return UpdateOutcome::kSkipped;
    }

    // Horizontal normal to the road. The road's own direction is deliberately
    // NOT measured -- see the header.
    const Vec3 direction = match.segment_direction;
    const double horizontal = std::hypot(direction.x(), direction.y());
    if (horizontal < 1e-9) {
        ++diagnostics_.map_match_skipped;
        return UpdateOutcome::kSkipped;
    }
    const Vec3 normal(-direction.y() / horizontal, direction.x() / horizontal, 0.0);

    // Scalar measurement: the component of position along the road normal
    // should equal that of the snapped point. Linear in the error state, so
    // this takes the cheap path like GNSS.
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, kStateDim);
    H.block<1, 3>(0, kPositionIndex) = normal.transpose();

    Eigen::VectorXd innovation(1);
    innovation(0) = normal.dot(match.snapped_position - nominal_.position);

    Eigen::MatrixXd sqrt_R(1, 1);
    sqrt_R(0, 0) = p.cross_track_sigma_m;

    const UpdateOutcome outcome =
        updateLinear(H, innovation, sqrt_R, p.gate_confidence);
    if (outcome == UpdateOutcome::kApplied) {
        ++diagnostics_.map_match_applied;
    } else if (outcome == UpdateOutcome::kRejectedByGate) {
        ++diagnostics_.map_match_rejected;
    }
    return outcome;
}

// --- Non-holonomic constraint ----------------------------------------------

UkfFusionEngine::UpdateOutcome UkfFusionEngine::updateNonHolonomic() {
    const auto& p = config_.non_holonomic;

    // Below the speed gate the constraint is not weak, it is wrong: a
    // stationary vehicle satisfies "zero lateral velocity" for every possible
    // heading, so the measurement carries no information but the filter would
    // still act on its noise.
    if (nominal_.velocity.norm() < p.min_speed_mps) {
        ++diagnostics_.nhc_skipped;
        return UpdateOutcome::kSkipped;
    }

    // Lateral and vertical components of body-frame velocity, both asserted
    // to be zero.
    const MeasurementFunction h = [](const NavState& s) -> Eigen::VectorXd {
        return bodyVelocity(s).segment<2>(1);
    };

    Eigen::VectorXd sqrt_r_diagonal(2);
    sqrt_r_diagonal << p.lateral_sigma_mps, p.vertical_sigma_mps;

    const UpdateOutcome outcome =
        updateUnscented(h, Eigen::Vector2d::Zero(),
                        Eigen::MatrixXd(sqrt_r_diagonal.asDiagonal()),
                        p.gate_confidence);
    if (outcome == UpdateOutcome::kApplied) {
        ++diagnostics_.nhc_applied;
    } else if (outcome == UpdateOutcome::kRejectedByGate) {
        ++diagnostics_.nhc_rejected;
    }
    return outcome;
}

// --- Learned velocity model --------------------------------------------

UkfFusionEngine::UpdateOutcome UkfFusionEngine::updateVelocityModel(
    double predicted_forward_speed_mps) {
    const auto& p = config_.velocity_model;

    // Correct the measured systematic bias before this is treated as an
    // unbiased measurement.
    const double corrected_speed_mps = predicted_forward_speed_mps - p.bias_mps;

    // Forward component of body-frame velocity -- the axis updateNonHolonomic
    // deliberately leaves unconstrained. Nonlinear in the attitude error, so
    // this takes the unscented path.
    const MeasurementFunction h = [](const NavState& s) -> Eigen::VectorXd {
        Eigen::VectorXd z(1);
        z(0) = bodyVelocity(s).x();
        return z;
    };

    Eigen::VectorXd z(1);
    z(0) = corrected_speed_mps;

    Eigen::MatrixXd sqrt_R(1, 1);
    sqrt_R(0, 0) = p.sigma_mps * p.correlation_inflation;

    const UpdateOutcome outcome =
        updateUnscented(h, z, sqrt_R, p.gate_confidence);
    if (outcome == UpdateOutcome::kApplied) {
        ++diagnostics_.velocity_model_applied;
    } else if (outcome == UpdateOutcome::kRejectedByGate) {
        ++diagnostics_.velocity_model_rejected;
    }
    return outcome;
}

}  // namespace driftless
