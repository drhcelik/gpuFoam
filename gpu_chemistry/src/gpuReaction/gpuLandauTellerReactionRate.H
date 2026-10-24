#pragma once

#include <cmath>

#include "device_array.H"
#include "gpu_constants.H"
#include "mdspan.H"

namespace FoamGpu {

///
///@brief Landau-Teller reaction rate, as OpenFOAM-dev
/// LandauTellerReactionRate:
///
///     k = A*T^beta*exp(-Ta/T + B/T^(1/3) + C/T^(2/3))
///
class gpuLandauTellerReactionRate {

    gScalar A_;
    gScalar beta_;
    gScalar Ta_;
    gScalar B_;
    gScalar C_;

public:
    gpuLandauTellerReactionRate() = default;

    inline gpuLandauTellerReactionRate(const gScalar A,
                                       const gScalar beta,
                                       const gScalar Ta,
                                       const gScalar B,
                                       const gScalar C)
        : A_(A)
        , beta_(beta)
        , Ta_(Ta)
        , B_(B)
        , C_(C) {}

    ///
    ///@brief Computes the the reaction rate.
    ///
    inline CUDA_HOSTDEV gScalar
    operator()(const gScalar                  p,
               const gScalar                  T,
               const mdspan<const gScalar, 1> c) const {
        (void)p;
        (void)c;
        const gScalar cbrtT = cbrt(T);
        return A_ * exp(beta_ * log(T) - Ta_ / T + B_ / cbrtT +
                        C_ / (cbrtT * cbrtT));
    }

    ///
    ///@brief Computes the derivative of the reaction rate w.r.t
    /// temperature.
    ///
    inline CUDA_HOSTDEV gScalar
    ddT(const gScalar                  p,
        const gScalar                  T,
        const mdspan<const gScalar, 1> c) const {
        const gScalar cbrtT = cbrt(T);
        const gScalar k     = this->operator()(p, T, c);
        const gScalar deriv = beta_ + Ta_ / T - B_ / (3 * cbrtT) -
                              2 * C_ / (3 * cbrtT * cbrtT);
        return k * deriv / T;
    }

    ///
    ///@brief Determines if the rate is a function of concentration.
    ///
    ///@return False.
    ///
    inline CUDA_HOSTDEV bool hasDdc() const { return false; }

    ///
    ///@brief The rate is not a function of concentration.
    ///
    inline CUDA_HOSTDEV void ddc(const gScalar                  p,
                                 const gScalar                  T,
                                 const mdspan<const gScalar, 1> c,
                                 mdspan<gScalar, 1> ddc) const {}

    ///
    ///@brief Computes the reaction rate and its derivative w.r.t
    /// temperature.
    ///
    inline CUDA_HOSTDEV device_array<gScalar, 2>
                        everything(const gScalar                  p,
                                   const gScalar                  T,
                                   const mdspan<const gScalar, 1> c,
                                   mdspan<gScalar, 1>             ddc) const {
        const gScalar cbrtT = cbrt(T);
        const gScalar k     = this->operator()(p, T, c);
        const gScalar deriv = beta_ + Ta_ / T - B_ / (3 * cbrtT) -
                              2 * C_ / (3 * cbrtT * cbrtT);
        return device_array<gScalar, 2>({k, k * deriv / T});
    }
};

} // namespace FoamGpu
