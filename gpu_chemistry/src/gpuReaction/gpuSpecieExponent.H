#pragma once

#include "gpu_constants.H"

namespace FoamGpu {

/// @brief Return true if N is an integer, using the same rule as the
/// OpenFOAM-dev specieExponent(scalar) constructor: floor(N) == N.
inline CUDA_HOSTDEV bool isInteger(gScalar N) { return floor(N) == N; }

inline CUDA_HOSTDEV gScalar integerPow(const gScalar x,
                                       const gLabel  e) {
    if (e == 0) { return 1; }

    gScalar xx = e > 0 ? x : 1 / x;

    gScalar y = 1;

    for (gLabel i = e; i != 0; i /= 2) {
        if (i % 2) { y *= xx; }

        xx *= xx;
    }

    return y;
}

struct gpuSpecieExponent {

    gLabel noIntegerExponent_ = gpuLabelMax;

    gLabel integerExponent_;

    gScalar scalarExponent_;

    inline CUDA_HOSTDEV gpuSpecieExponent()
        : integerExponent_(gpuLabelMax)
        , scalarExponent_(gpuScalarNaN) {}

    inline CUDA_HOSTDEV
    gpuSpecieExponent(const gLabel integerExponent)
        : integerExponent_(integerExponent)
        , scalarExponent_(integerExponent) {}

    inline CUDA_HOSTDEV
    gpuSpecieExponent(const gScalar scalarExponent)
        : integerExponent_(gpuLabelMax)
        , scalarExponent_(scalarExponent) {}

    inline CUDA_HOSTDEV bool hasIntegerExponent() const {
        return integerExponent_ != noIntegerExponent_;
    }

    inline CUDA_HOSTDEV operator gScalar() const {
        return scalarExponent_;
    }

    inline CUDA_HOSTDEV gpuSpecieExponent&
    operator=(const gLabel integerExponent) {
        return *this = gpuSpecieExponent(integerExponent);
    }

    inline CUDA_HOSTDEV gpuSpecieExponent&
    operator=(const gScalar scalarExponent) {
        return *this = gpuSpecieExponent(scalarExponent);
    }

    inline CUDA_HOSTDEV gpuSpecieExponent operator-() const {
        if (hasIntegerExponent()) {
            return -integerExponent_;
        } else {
            return -scalarExponent_;
        }
    }

    inline CUDA_HOSTDEV friend gpuSpecieExponent
    operator+(const gpuSpecieExponent& a,
              const gpuSpecieExponent& b) {
        if (a.hasIntegerExponent() && b.hasIntegerExponent()) {
            return a.integerExponent_ + b.integerExponent_;
        } else {
            return a.scalarExponent_ + b.scalarExponent_;
        }
    }

    inline CUDA_HOSTDEV friend gpuSpecieExponent
    operator-(const gpuSpecieExponent& a,
              const gpuSpecieExponent& b) {
        return a + (-b);
    }
};

inline CUDA_HOSTDEV gScalar speciePow(const gScalar            x,
                                      const gpuSpecieExponent& e) {
    if (e.hasIntegerExponent()) {
        return integerPow(x, e.integerExponent_);
    } else {
        return pow(x, e.scalarExponent_);
    }
}

/// @brief Derivative of speciePow(c, e) w.r.t. c, mirroring
/// Foam::dpowdc (OpenFOAM-dev specieExponentI.H).  A zero exponent gives
/// 0, and exponents below 1 are evaluated at max(c, small) to avoid the
/// singularity at c = 0, where small = gpuSmall = OpenFOAM's
/// doubleScalarSmall (machine epsilon).
inline CUDA_HOSTDEV gScalar dpowdc(const gScalar            c,
                                   const gpuSpecieExponent& e) {
    if (e.hasIntegerExponent()) {
        if (e.integerExponent_ >= 1) {
            return e.integerExponent_ *
                   integerPow(c, e.integerExponent_ - 1);
        } else // (e.integerExponent_ == 0)
        {
            return 0;
        }
    } else {
        if (e.scalarExponent_ >= 1) {
            return e.scalarExponent_ * pow(c, e.scalarExponent_ - 1);
        } else if (e.scalarExponent_ == 0) {
            return 0;
        } else {
            return e.scalarExponent_ *
                   pow(fmax(c, gpuSmall), e.scalarExponent_ - 1);
        }
    }
}

} // namespace FoamGpu
