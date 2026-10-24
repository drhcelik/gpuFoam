#pragma once

#include <cmath>

#include "device_array.H"
#include "gpu_constants.H"
#include "mdspan.H"

namespace FoamGpu {

///
///@brief Janev, Langer, Evans and Post reaction rate, as OpenFOAM-dev
/// JanevReactionRate:
///
///     k = A*T^beta*exp(-Ta/T)*exp(sum_n b_n*log(T)^n), n = 0..8
///
class gpuJanevReactionRate {

public:
    static constexpr gLabel nb_ = 9;

    using coeffArray = device_array<gScalar, nb_>;

private:
    gScalar    A_;
    gScalar    beta_;
    gScalar    Ta_;
    coeffArray b_;

public:
    gpuJanevReactionRate() = default;

    inline gpuJanevReactionRate(const gScalar     A,
                                const gScalar     beta,
                                const gScalar     Ta,
                                const coeffArray& b)
        : A_(A)
        , beta_(beta)
        , Ta_(Ta)
        , b_(b) {}

    ///
    ///@brief Computes the the reaction rate.
    ///
    inline CUDA_HOSTDEV gScalar
    operator()(const gScalar                  p,
               const gScalar                  T,
               const mdspan<const gScalar, 1> c) const {
        (void)p;
        (void)c;
        const gScalar lnT    = log(T);
        gScalar       expArg = beta_ * lnT - Ta_ / T + b_[0] + b_[1] * lnT;
        gScalar       lnTn   = lnT;
        for (gLabel n = 2; n < nb_; n++) {
            lnTn *= lnT;
            expArg += b_[n] * lnTn;
        }
        return A_ * exp(expArg);
    }

    ///
    ///@brief Computes the derivative of the reaction rate w.r.t
    /// temperature.
    ///
    inline CUDA_HOSTDEV gScalar
    ddT(const gScalar                  p,
        const gScalar                  T,
        const mdspan<const gScalar, 1> c) const {
        (void)p;
        (void)c;
        const gScalar lnT    = log(T);
        gScalar       expArg = beta_ * lnT - Ta_ / T + b_[0] + b_[1] * lnT;
        gScalar       deriv  = beta_ + Ta_ / T + b_[1];
        gScalar       lnTn   = lnT;
        for (gLabel n = 2; n < nb_; n++) {
            deriv += n * b_[n] * lnTn;
            lnTn *= lnT;
            expArg += b_[n] * lnTn;
        }
        return A_ * exp(expArg) * deriv / T;
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
        (void)p;
        (void)c;
        const gScalar lnT    = log(T);
        gScalar       expArg = beta_ * lnT - Ta_ / T + b_[0] + b_[1] * lnT;
        gScalar       deriv  = beta_ + Ta_ / T + b_[1];
        gScalar       lnTn   = lnT;
        for (gLabel n = 2; n < nb_; n++) {
            deriv += n * b_[n] * lnTn;
            lnTn *= lnT;
            expArg += b_[n] * lnTn;
        }
        const gScalar k = A_ * exp(expArg);
        return device_array<gScalar, 2>({k, k * deriv / T});
    }
};

} // namespace FoamGpu
