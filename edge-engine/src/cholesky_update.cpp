#include "driftless/cholesky_update.h"

#include <Eigen/QR>

#include <cmath>

namespace driftless::linalg {

bool cholUpdate(Eigen::Ref<Eigen::MatrixXd> S,
                const Eigen::Ref<const Eigen::VectorXd>& u, double sigma) {
    const Eigen::Index n = S.rows();
    if (S.cols() != n || u.size() != n) return false;
    if (sigma == 0.0) return true;

    // Fold |sigma| into the vector so the loop below only has to deal with a
    // sign. (A + sigma*u*u^T) == (A + sign * w*w^T) with w = sqrt(|sigma|)*u.
    const double sign = sigma > 0.0 ? 1.0 : -1.0;
    Eigen::VectorXd w = std::sqrt(std::abs(sigma)) * u;

    // Golub's rank-1 modification, applied column by column. Each step rotates
    // the remaining trailing vector into the factor, so w is consumed as we go.
    for (Eigen::Index k = 0; k < n; ++k) {
        const double d = S(k, k);
        if (!(d > 0.0)) return false;

        const double r_squared = d * d + sign * w(k) * w(k);
        // A downdate can take the matrix out of the positive-definite cone.
        // That is a real condition the caller has to handle, not an error.
        if (!(r_squared > 0.0)) return false;

        const double r = std::sqrt(r_squared);
        const double c = r / d;
        const double s = w(k) / d;
        S(k, k) = r;

        for (Eigen::Index i = k + 1; i < n; ++i) {
            S(i, k) = (S(i, k) + sign * s * w(i)) / c;
            w(i) = c * w(i) - s * S(i, k);
        }
    }
    return true;
}

Eigen::MatrixXd qrToLowerTriangular(const Eigen::Ref<const Eigen::MatrixXd>& A) {
    const Eigen::Index n = A.cols();
    // A = Q*R with R upper triangular, so A^T*A == R^T*R. The lower-triangular
    // factor we want is therefore just R^T -- A^T*A is never formed, which is
    // the whole point: it would square the condition number.
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::MatrixXd R = qr.matrixQR().topRows(n).triangularView<Eigen::Upper>();

    // The QR is only unique up to the sign of each row of R. Normalize to a
    // non-negative diagonal so the factor is deterministic and comparable to
    // what an LLT of A^T*A would produce.
    for (Eigen::Index i = 0; i < n; ++i) {
        if (R(i, i) < 0.0) R.row(i) = -R.row(i);
    }
    return R.transpose();
}

}  // namespace driftless::linalg
