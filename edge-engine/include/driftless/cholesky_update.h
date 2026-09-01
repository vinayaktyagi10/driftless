#pragma once

#include <Eigen/Core>

namespace driftless::linalg {

// Rank-1 Cholesky update/downdate.
//
// Given a lower-triangular S with A = S*S^T, overwrite S in place with the
// lower-triangular Cholesky factor of  A + sigma * u * u^T.
//
// sigma may be any real number; its sign selects update (>0) or downdate (<0)
// and its magnitude is folded into u. A downdate can legitimately fail — it is
// subtracting from the covariance, and nothing guarantees the result is still
// positive definite — so this returns false rather than producing a factor with
// a NaN or negative diagonal. Callers must check.
//
// This exists because Eigen's LLT::rankUpdate cannot be seeded with an
// already-computed factor; it only updates a factorization it computed itself.
[[nodiscard]] bool cholUpdate(Eigen::Ref<Eigen::MatrixXd> S,
                              const Eigen::Ref<const Eigen::VectorXd>& u,
                              double sigma);

// Lower-triangular S with S*S^T == A^T * A, obtained from the QR of A without
// ever forming A^T*A. This is how the SR-UKF gets its propagated factor: stack
// the weighted sigma deviations on top of sqrt(Q) and hand the result here.
//
// A must have at least as many rows as columns. The returned factor is
// normalized to a non-negative diagonal so the result is unique.
[[nodiscard]] Eigen::MatrixXd qrToLowerTriangular(
    const Eigen::Ref<const Eigen::MatrixXd>& A);

}  // namespace driftless::linalg
