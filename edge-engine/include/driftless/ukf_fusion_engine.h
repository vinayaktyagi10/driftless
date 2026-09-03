#pragma once

#include "driftless/hmm_map_matcher.h"
#include "driftless/imu_noise.h"
#include "driftless/so3.h"
#include "driftless/types.h"

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <functional>
#include <optional>

namespace driftless {

// Error-state square-root Unscented Kalman Filter, retuned for ~200Hz FOG IMU
// input instead of ~100Hz consumer MEMS.
//
// This is the reference implementation of the filter; the Kotlin engine in
// android/.../fusion/UkfFusionEngine.kt is a port of it, not the other way
// round. State layout, frame conventions (types.h) and the mechanization below
// must stay in sync across the two even though they share no runtime.
//
// TWO STATES, and the distinction is the whole design:
//
//   The NOMINAL state [p, v, q, b_a, b_g] is integrated directly and is what
//   callers read. It is NOT what the covariance is over.
//
//   The ERROR state [dp, dv, dtheta, db_a, db_g] (n = 15) is what the filter
//   estimates, and it is held at zero between steps. Attitude has no global
//   3-parameter representation: a filter carrying heading as a plain angle
//   averages sigma points across the +/-pi branch cut and produces a garbage
//   mean, and a filter carrying a quaternion in the covariance needs a
//   manifold mean because the weighted average of quaternions is not a
//   quaternion. Carrying the error instead means that after the decompose step
//   every sigma point is an ordinary tangent-space vector, so the mean is a
//   plain weighted sum.
//
// The covariance is carried as its lower-triangular Cholesky factor S, never
// as P. A blackout at 200Hz is ~12,000 predict steps with no update to
// re-condition anything; propagating S keeps positive-definiteness structural
// instead of leaving it to accumulate asymmetry until a factorization fails in
// the middle of exactly the scenario being scored.
class UkfFusionEngine {
public:
    static constexpr int kStateDim = 15;
    static constexpr int kSigmaPoints = 2 * kStateDim + 1;

    using ErrorVector = Eigen::Matrix<double, kStateDim, 1>;
    using ErrorMatrix = Eigen::Matrix<double, kStateDim, kStateDim>;

    // Index of each block within the error vector.
    static constexpr int kPositionIndex = 0;
    static constexpr int kVelocityIndex = 3;
    static constexpr int kAttitudeIndex = 6;
    static constexpr int kAccelBiasIndex = 9;
    static constexpr int kGyroBiasIndex = 12;

    // What became of a measurement. Rejection is a normal, expected outcome --
    // it is the point of the gate -- so it is reported rather than thrown, and
    // it is distinguished from "the filter broke".
    enum class UpdateOutcome {
        kApplied,
        kRejectedByGate,     // innovation failed the chi-squared test
        kSkipped,            // preconditions not met (e.g. NHC below speed gate)
        kNumericalFailure,   // covariance factorization gave up
    };

    struct GnssParams {
        // Receivers under-report their own error in exactly the situation that
        // matters. These floors stop a fix claiming implausible precision.
        double min_horizontal_accuracy_m = 1.0;
        double min_vertical_accuracy_m = 2.0;
        double min_speed_accuracy_mps = 0.1;

        // Reported accuracy is computed from satellite geometry assuming
        // line-of-sight. In an urban canyon that assumption is false, and the
        // receiver has no way to know -- which is how a 40 m multipath fix
        // arrives claiming 5 m accuracy. Satellite count is a crude but
        // genuinely independent proxy for whether the assumption holds, so R is
        // inflated as the count drops below what open sky would give.
        int healthy_satellite_count = 8;
        double satellite_deficit_inflation = 0.25;

        double gate_confidence = 0.99;
    };

    struct NonHolonomicParams {
        // A road vehicle does not slide sideways or leave the road surface, so
        // body-frame lateral and vertical velocity are ~0. These sigmas are not
        // sensor noise -- they are how wrong the constraint itself is allowed
        // to be, covering suspension travel, camber, and mild slip.
        //
        // Sized generously on purpose. The constraint is applied every time it
        // is called, but consecutive lateral-velocity errors are almost
        // perfectly correlated in time -- suspension lean and road camber do
        // not resample themselves at 10Hz. Treating each application as an
        // independent observation therefore injects far more information than
        // actually arrived. Measured consequence of sigma=0.1 at 10Hz: the
        // attitude covariance collapses to roughly 1 degree, and the filter
        // then REJECTS 21 of the honest GNSS fixes that follow the blackout,
        // because they disagree with an estimate it has become unjustifiably
        // certain of. A larger sigma is the cheap defence; the principled one
        // would be to model the time correlation.
        double lateral_sigma_mps = 0.3;
        double vertical_sigma_mps = 0.3;

        // Below this speed the constraint is not merely weak but WRONG: a
        // stationary vehicle has zero velocity in every direction, so the
        // constraint carries no information about heading, and applying it
        // anyway drives the attitude estimate toward whatever the noise says.
        // It is also violated outright during a skid.
        double min_speed_mps = 2.0;

        double gate_confidence = 0.99;
    };

    struct VelocityModelParams {
        // Held-out residuals of the learned speed/heading model, pooled across
        // BOTH held-out routes (training/artifacts/metrics/measurement_noise.md).
        // A single-route figure (2.2046) was ~30% optimistic.
        double sigma_mps = 2.893;
        // Not a sensor offset -- it changes sign between held-out routes
        // (+0.4853 on one, -0.5536 on the other), so it is a property of each
        // route's speed distribution interacting with a shrinkage estimator,
        // not something safe to correct for at inference time. Left at 0.0 and
        // kept as a field for future per-deployment calibration only.
        double bias_mps = 0.0;

        // Consecutive predictions share most of their 8s context, so their
        // errors are NOT independent -- lag-1 autocorrelation 0.7367,
        // decorrelation time 8s against a 2s update interval. Feeding each one
        // in as though independent over-informs the filter by about 2x in
        // sigma. Same trap as NonHolonomicParams::lateral_sigma_mps above (an
        // over-tight sigma at high rate collapsed the attitude covariance and
        // cost 21 honest GNSS fixes); the same cheap defence is used here
        // rather than gating the update rate to the decorrelation time.
        double correlation_inflation = 2.0;

        double gate_confidence = 0.99;
    };

    struct MapMatchParams {
        // How far the true position may sit from the mapped centreline: half a
        // carriageway plus map digitisation error. This is a property of roads
        // and of the map, NOT of how well the match went.
        double cross_track_sigma_m = 2.5;
        // Beyond this the match is not trustworthy enough to act on at all.
        double max_distance_to_road_m = 30.0;
        double gate_confidence = 0.99;
    };

    struct Diagnostics {
        double last_nis = 0.0;
        int gnss_applied = 0;
        int gnss_rejected = 0;
        int nhc_applied = 0;
        int nhc_rejected = 0;
        int nhc_skipped = 0;
        int map_match_applied = 0;
        int map_match_rejected = 0;
        int map_match_skipped = 0;
        int velocity_model_applied = 0;
        int velocity_model_rejected = 0;
    };

    struct Config {
        // Scaled unscented transform parameters (van der Merwe). alpha sets how
        // far the sigma points spread; beta = 2 is optimal for a Gaussian prior.
        // With kappa = 0 and alpha < 1 the centre weight W0c is negative, which
        // is exactly why the covariance step needs a Cholesky *downdate* and
        // not just an update.
        double alpha = 1e-3;
        double beta = 2.0;
        double kappa = 0.0;

        Vec3 gravity_ned = Vec3(0.0, 0.0, kGravity);

        // Samples further apart than this are treated as a gap rather than a
        // step: integrating a long interval as if it were one IMU sample
        // produces a confidently wrong answer, which is worse than a gap.
        double max_step_seconds = 0.1;

        GnssParams gnss;
        NonHolonomicParams non_holonomic;
        MapMatchParams map_match;
        VelocityModelParams velocity_model;
    };

    UkfFusionEngine(const NavState& initial_state,
                    const ErrorMatrix& initial_covariance_sqrt,
                    const ImuNoiseParams& noise, const Config& config);

    // Config's default member initializers are not usable as a default
    // argument here -- the enclosing class is still incomplete at that point --
    // so the defaulted form is a separate overload rather than `= Config{}`.
    UkfFusionEngine(const NavState& initial_state,
                    const ErrorMatrix& initial_covariance_sqrt,
                    const ImuNoiseParams& noise);

    // Propagate to this sample's timestamp. Returns false if the step was
    // rejected (non-monotonic or over-long timestamp) or if the covariance
    // factorization failed -- both leave the filter unchanged, and both are
    // conditions the caller should surface rather than swallow.
    bool predict(const ImuSample& sample);

    // --- Measurement updates ------------------------------------------------
    //
    // GNSS is handled by a LINEAR square-root update, not an unscented one, and
    // that is deliberate: in error-state coordinates the measurement model for
    // a position fix is h(dx) = dp, which is exactly linear. Pushing 31 sigma
    // points through an identity map would burn the arithmetic to arrive at
    // precisely the answer the linear update gives in closed form -- and would
    // arrive at it only approximately, because the unscented transform is exact
    // for linear maps but its square-root implementation still rounds.
    //
    // The non-holonomic constraint gets the unscented path instead, because its
    // model h(x) = (R(q)^T * v)[lateral, vertical] genuinely is nonlinear in
    // the attitude error -- which is the whole reason the constraint is
    // informative about heading in the first place.
    UpdateOutcome updateGnss(const GnssFix& fix);
    UpdateOutcome updateNonHolonomic();

    // Forward-speed measurement from the learned speed/heading model (training
    // handover, see VelocityModelParams). Fits the one axis the non-holonomic
    // constraint deliberately leaves free -- see bodyVelocity -- so it takes
    // the unscented path for the same reason NHC does: h is nonlinear in the
    // attitude error, which is exactly why this is informative about heading.
    UpdateOutcome updateVelocityModel(double predicted_forward_speed_mps);

    // Fold a map match back into the estimate.
    //
    // CROSS-TRACK ONLY, and that restriction is the whole design. The snapped
    // position was produced BY matching this filter's own position estimate
    // against the map, so treating it as an independent observation of position
    // would be the filter learning from itself -- the covariance would shrink
    // on information it already had, the estimate would become overconfident,
    // and a wrong match would then be impossible to escape because every honest
    // GNSS fix that disagreed would fail the gate.
    //
    // What the map genuinely adds is the constraint "the vehicle is on a road",
    // which is information about the direction PERPENDICULAR to the road and
    // nothing else. Along the road the snapped position carries no information
    // the filter did not already supply. So the update is a single scalar
    // measurement along the road normal, and the along-track component is
    // discarded.
    //
    // This does not eliminate the feedback loop -- a confidently wrong match
    // still pulls sideways -- it bounds it to one dimension and sizes its noise
    // by road geometry rather than by match quality. The remaining defence is
    // the matcher's own transition term refusing to jump between roads.
    UpdateOutcome updateMapMatch(const MapMatchResult& match);

    // Generic linear update: innovation nu = z - h(nominal), model H, noise
    // factor sqrt_R. Gated on NIS at the given confidence.
    UpdateOutcome updateLinear(const Eigen::MatrixXd& H,
                               const Eigen::VectorXd& innovation,
                               const Eigen::MatrixXd& sqrt_R,
                               double gate_confidence);

    // Generic unscented update for a nonlinear measurement function evaluated
    // on a full NavState.
    using MeasurementFunction = std::function<Eigen::VectorXd(const NavState&)>;
    UpdateOutcome updateUnscented(const MeasurementFunction& h,
                                  const Eigen::VectorXd& z,
                                  const Eigen::MatrixXd& sqrt_R,
                                  double gate_confidence);

    [[nodiscard]] const Diagnostics& diagnostics() const { return diagnostics_; }

    [[nodiscard]] const NavState& state() const { return nominal_; }
    [[nodiscard]] const ErrorMatrix& covarianceSqrt() const { return sqrt_covariance_; }
    // Convenience for tests and telemetry only; the filter never forms P.
    [[nodiscard]] ErrorMatrix covariance() const {
        return sqrt_covariance_ * sqrt_covariance_.transpose();
    }

    // Sigma points for a zero-mean error with square-root covariance S.
    // Exposed rather than buried in predict() because the moment-matching
    // identities it has to satisfy are exact, so they can be tested directly
    // and independently of any dynamics -- and because the update step will
    // need the same set.
    [[nodiscard]] static std::array<ErrorVector, kSigmaPoints> sigmaPoints(
        const ErrorMatrix& sqrt_covariance, double gamma);

    // Unscented-transform parameters derived from Config, exposed for the same
    // reason.
    [[nodiscard]] double gamma() const { return gamma_; }
    [[nodiscard]] double weightMean0() const { return weight_mean_0_; }
    [[nodiscard]] double weightCov0() const { return weight_cov_0_; }
    [[nodiscard]] double weightI() const { return weight_i_; }

    // Strapdown mechanization for a single interval. Exposed because the
    // trajectory simulator inverts it to generate ideal IMU samples, and a
    // simulator that shares this function would hide sign errors rather than
    // expose them -- so tests use it only as the forward reference.
    [[nodiscard]] static NavState mechanize(const NavState& s, const Vec3& accel,
                                            const Vec3& gyro, double dt,
                                            const Vec3& gravity_ned);

    // Compose an error vector onto a nominal state, and its inverse.
    [[nodiscard]] static NavState compose(const NavState& nominal,
                                          const ErrorVector& error);
    [[nodiscard]] static ErrorVector decompose(const NavState& s,
                                               const NavState& nominal);

    // Body-frame velocity, i.e. the quantity the non-holonomic constraint is
    // asserted about. Exposed because it is what a test needs to check the
    // constraint is doing anything.
    [[nodiscard]] static Vec3 bodyVelocity(const NavState& s) {
        return s.orientation.conjugate() * s.velocity;
    }

private:
    void computeWeights();
    // Measurement noise factor for a fix, after flooring the reported accuracy
    // and inflating for satellite geometry.
    [[nodiscard]] Eigen::MatrixXd gnssSqrtNoise(const GnssFix& fix,
                                                bool use_velocity) const;

    NavState nominal_;
    ErrorMatrix sqrt_covariance_;
    ImuNoiseParams noise_;
    Config config_;

    std::optional<std::int64_t> last_timestamp_nanos_;
    Diagnostics diagnostics_;

    double gamma_ = 0.0;      // sqrt(n + lambda), the sigma-point spread
    double weight_mean_0_ = 0.0;
    double weight_cov_0_ = 0.0;
    double weight_i_ = 0.0;   // shared by all 2n outer points, always positive
};

}  // namespace driftless
