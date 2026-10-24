#pragma once

#include "cuda_host_dev.H"
#include "gpu_constants.H"
#include "swap.H"

namespace FoamGpu {

static inline CUDA_HOSTDEV void
LUDecompose(mdspan<gScalar, 2> matrix,
            mdspan<gLabel, 1>  pivotIndices,
            mdspan<gScalar, 1> vv) {

    const gLabel m = static_cast<gLabel>(matrix.extents().extent(1));

    // gLabel sign = 1;
    for (gLabel i = 0; i < m; i++) {
        gScalar largestCoeff = 0.0;
        gScalar temp;
        const gScalar* __restrict__ matrixi = &(matrix(i, 0));
        for (gLabel j = 0; j < m; j++) {
            if ((temp = fabs(matrixi[j])) > largestCoeff)
            {
                largestCoeff = temp;
            }
        }

        vv[i] = 1.0 / largestCoeff;
    }

    // std::array<gScalr, SPECIE_MAX> matrix_i{};

    for (gLabel j = 0; j < m; j++) {

        gScalar* __restrict__ matrixj = &(matrix(j, 0));


        for (gLabel i = 0; i < j; i++) {
            gScalar* __restrict__ matrixi = &(matrix(i, 0));

            gScalar sum = matrixi[j];
            for (gLabel k = 0; k < i; k++) {
                sum -= matrixi[k] * matrix(k, j);
            }
            matrixi[j] = sum;
        }

        gLabel iMax = 0;

        gScalar largestCoeff = 0.0;
        for (gLabel i = j; i < m; i++) {

            gScalar* __restrict__ matrixi = &(matrix(i, 0));
            gScalar sum = matrixi[j];
            for (gLabel k = 0; k < j; k++) {
                sum -= matrixi[k] * matrix(k, j);
            }
            matrixi[j] = sum;

            gScalar temp;

            if ((temp = vv[i]*fabs(sum)) >= largestCoeff)
            {
                largestCoeff = temp;
                iMax = i;
            }
        }

        pivotIndices[j] = iMax;

        if (j != iMax) {
            gScalar* __restrict__ matrixiMax = &(matrix(iMax, 0));
            for (gLabel k = 0; k < m; k++) {
                swap(matrixj[k], matrixiMax[k]);
            }

            // sign *= -1;
            vv[iMax] = vv[j];
        }


        if (matrixj[j] == 0.0) { matrixj[j] = gpuSmall; }

        if (j != m - 1) {
            gScalar rDiag = 1.0 / matrixj[j];

            for (gLabel i = j + 1; i < m; i++) {
                matrix(i, j) *= rDiag;
            }
        }
    }
}

static inline CUDA_HOSTDEV void
LUBacksubstitute(const mdspan<const gScalar, 2> luMatrix,
                 const mdspan<const gLabel, 1>  pivotIndices,
                 mdspan<gScalar, 1>             sourceSol) {

    const gLabel m =
        static_cast<gLabel>(luMatrix.extents().extent(1));
    gLabel ii = 0;

    for (gLabel i = 0; i < m; i++) {
        gLabel  ip    = pivotIndices[i];
        gScalar sum   = sourceSol[ip];
        sourceSol[ip] = sourceSol[i];

        const gScalar* __restrict__ luMatrixi = &(luMatrix(i, 0));

        if (ii != 0) {
            for (gLabel j = ii - 1; j < i; j++) {
                sum -= luMatrixi[j] * sourceSol[j];
            }
        } else if (sum != gScalar(0)) {
            ii = i + 1;
        }

        sourceSol[i] = sum;
    }

    for (gLabel i = m - 1; i >= 0; i--) {
        gScalar sum = sourceSol[i];
        const gScalar* __restrict__ luMatrixi = &(luMatrix(i, 0));
        for (gLabel j = i + 1; j < m; j++) {
            sum -= luMatrixi[j] * sourceSol[j];
        }

        sourceSol[i] = sum / luMatrixi[i];
    }
}

/// @brief LU decomposition without pivoting in the ikj-loop form of the
/// OpenFOAM-dev LUDecompose(scalarSquareMatrix&, const scalar rowTol)
/// (5e2f2eafea, 10f026fd67).  Row multipliers with magnitude <= rowTol are
/// set to zero and the corresponding row update is skipped.
/// @param matrix The matrix to decompose in place.
/// @param rowTol The row multiplier tolerance.
static inline CUDA_HOSTDEV void
LUDecompose(mdspan<gScalar, 2> matrix, const gScalar rowTol) {

    const gLabel m = static_cast<gLabel>(matrix.extents().extent(1));

    for (gLabel i = 0; i < m; i++) {
        gScalar* __restrict__ matrixi = &(matrix(i, 0));

        for (gLabel k = 0; k < i; k++) {
            const gScalar* __restrict__ matrixk = &(matrix(k, 0));

            // Compute the row multiplier for the lower triangular matrix L
            matrixi[k] /= matrixk[k];

            // If the row multiplier is 0 skip the inner j loop
            if (fabs(matrixi[k]) <= rowTol) {
                matrixi[k] = 0;
                continue;
            }

            // Inner 'j' loop updates the remainder of row 'i'
            for (gLabel j = k + 1; j < m; j++) {
                matrixi[j] -= matrixi[k] * matrixk[j];
            }
        }
    }
}

/// @brief Back-substitution for the unpivoted LU decomposition as the
/// OpenFOAM-dev LUBacksubstitute(const scalarSquareMatrix&, List<Type>&).
/// @param luMatrix The decomposed matrix.
/// @param sourceSol The source on input and the solution on output.
static inline CUDA_HOSTDEV void
LUBacksubstitute(const mdspan<const gScalar, 2> luMatrix,
                 mdspan<gScalar, 1>             sourceSol) {

    const gLabel m =
        static_cast<gLabel>(luMatrix.extents().extent(1));
    gLabel ii = 0;

    for (gLabel i = 0; i < m; i++) {
        gScalar sum = sourceSol[i];

        const gScalar* __restrict__ luMatrixi = &(luMatrix(i, 0));

        if (ii != 0) {
            for (gLabel j = ii - 1; j < i; j++) {
                sum -= luMatrixi[j] * sourceSol[j];
            }
        } else if (sum != gScalar(0)) {
            ii = i + 1;
        }

        sourceSol[i] = sum;
    }

    for (gLabel i = m - 1; i >= 0; i--) {
        gScalar sum = sourceSol[i];
        const gScalar* __restrict__ luMatrixi = &(luMatrix(i, 0));
        for (gLabel j = i + 1; j < m; j++) {
            sum -= luMatrixi[j] * sourceSol[j];
        }

        sourceSol[i] = sum / luMatrixi[i];
    }
}

} // namespace FoamGpu
