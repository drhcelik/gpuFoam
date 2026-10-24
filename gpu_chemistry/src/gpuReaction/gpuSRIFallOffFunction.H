#pragma once

#include <cmath>

#include "gpu_constants.H"

namespace FoamGpu {

///
///@brief The SRI fall-off function, as OpenFOAM-dev SRIFallOffFunction.
///
class gpuSRIFallOffFunction {

    gScalar a_, b_, c_, d_, e_;

public:
    gpuSRIFallOffFunction() = default;

    inline CUDA_HOSTDEV gpuSRIFallOffFunction(const gScalar a,
                                              const gScalar b,
                                              const gScalar c,
                                              const gScalar d,
                                              const gScalar e)
        : a_(a)
        , b_(b)
        , c_(c)
        , d_(d)
        , e_(e) {}

    inline CUDA_HOSTDEV gScalar operator()(const gScalar T,
                                           const gScalar Pr) const {
        const gScalar logPr = log10(fmax(Pr, gpuSmall));
        const gScalar X     = 1.0 / (1.0 + logPr * logPr);
        const gScalar psi   = a_ * exp(-b_ / T) + exp(-T / c_);
        return d_ * pow(psi, X) * pow(T, e_);
    }

    inline CUDA_HOSTDEV gScalar ddT(const gScalar T,
                                    const gScalar Pr,
                                    const gScalar F) const {
        const gScalar logPr = log10(fmax(Pr, gpuSmall));
        const gScalar X     = 1.0 / (1.0 + logPr * logPr);
        const gScalar psi   = a_ * exp(-b_ / T) + exp(-T / c_);
        const gScalar dpsidT =
            a_ * b_ / (T * T) * exp(-b_ / T) - 1.0 / c_ * exp(-T / c_);
        return F * (X / psi * dpsidT + e_ / T);
    }

    inline CUDA_HOSTDEV gScalar ddPr(const gScalar T,
                                     const gScalar Pr,
                                     const gScalar F) const {
        const gScalar logTen    = log(gScalar(10));
        const gScalar logPr     = log10(fmax(Pr, gpuSmall));
        const gScalar dlogPrdPr = Pr >= gpuSmall ? 1.0 / (logTen * Pr) : 0;
        const gScalar X         = 1.0 / (1.0 + logPr * logPr);
        const gScalar dXdPr     = -X * X * 2 * logPr * dlogPrdPr;
        const gScalar psi       = a_ * exp(-b_ / T) + exp(-T / c_);
        return F * log(psi) * dXdPr;
    }
};

} // namespace FoamGpu
