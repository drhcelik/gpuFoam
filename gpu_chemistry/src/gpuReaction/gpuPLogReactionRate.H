#pragma once

#include <cmath>

#include "device_array.H"
#include "gpu_constants.H"
#include "mdspan.H"

namespace FoamGpu {

///
///@brief Pressure-dependent reaction rate evaluated by logarithmic
/// interpolation of the Arrhenius rates of a table of pressures, as
/// OpenFOAM-dev pLogReactionRate. Up to nMax_ pressures are supported.
///
class gpuPLogReactionRate {

public:
    static constexpr gLabel nMax_ = 16;

    using coeffArray = device_array<gScalar, nMax_>;

private:
    gLabel     n_;
    coeffArray p_;
    coeffArray lnp_;
    coeffArray A_;
    coeffArray lnA_;
    coeffArray beta_;
    coeffArray Ta_;

    inline CUDA_HOSTDEV gScalar ki(const gLabel  i,
                                   const gScalar lnT,
                                   const gScalar T) const {
        return A_[i] * exp(beta_[i] * lnT - Ta_[i] / T);
    }

    inline CUDA_HOSTDEV gScalar lnki(const gLabel  i,
                                     const gScalar lnT,
                                     const gScalar T) const {
        return lnA_[i] + beta_[i] * lnT - Ta_[i] / T;
    }

    inline CUDA_HOSTDEV gScalar ddTi(const gLabel  i,
                                     const gScalar lnT,
                                     const gScalar T) const {
        return ki(i, lnT, T) * (beta_[i] + Ta_[i] / T) / T;
    }

    //- Index of the upper pressure of the interpolation interval
    inline CUDA_HOSTDEV gLabel upper(const gScalar p) const {
        gLabel pi = 1;
        while (p > p_[pi]) { pi++; }
        return pi;
    }

    inline CUDA_HOSTDEV gScalar fp(const gScalar lnp,
                                   const gLabel  pi) const {
        return (lnp - lnp_[pi - 1]) / (lnp_[pi] - lnp_[pi - 1]);
    }

public:
    gpuPLogReactionRate() = default;

    ///
    ///@brief Construct from the n pressures, in ascending order, and
    /// their Arrhenius coefficients.
    ///
    inline gpuPLogReactionRate(const gLabel      n,
                               const coeffArray& p,
                               const coeffArray& A,
                               const coeffArray& beta,
                               const coeffArray& Ta)
        : n_(n)
        , p_(p)
        , lnp_{}
        , A_(A)
        , lnA_{}
        , beta_(beta)
        , Ta_(Ta) {
        for (gLabel i = 0; i < n_; ++i) {
            lnp_[i] = log(p_[i]);
            lnA_[i] = log(A_[i]);
        }
    }

    ///
    ///@brief Computes the the reaction rate.
    ///
    inline CUDA_HOSTDEV gScalar
    operator()(const gScalar                  p,
               const gScalar                  T,
               const mdspan<const gScalar, 1> c) const {
        (void)c;
        const gScalar lnT = log(T);

        if (p <= p_[0]) { return ki(0, lnT, T); }
        if (p >= p_[n_ - 1]) { return ki(n_ - 1, lnT, T); }

        const gLabel  pi     = upper(p);
        const gScalar lnkim1 = lnki(pi - 1, lnT, T);
        const gScalar lnkii  = lnki(pi, lnT, T);
        return exp(lnkim1 + fp(log(p), pi) * (lnkii - lnkim1));
    }

    ///
    ///@brief Computes the derivative of the reaction rate w.r.t
    /// temperature.
    ///
    inline CUDA_HOSTDEV gScalar
    ddT(const gScalar                  p,
        const gScalar                  T,
        const mdspan<const gScalar, 1> c) const {
        (void)c;
        const gScalar lnT = log(T);

        if (p <= p_[0]) { return ddTi(0, lnT, T); }
        if (p >= p_[n_ - 1]) { return ddTi(n_ - 1, lnT, T); }

        const gLabel  pi     = upper(p);
        const gScalar lnkim1 = lnki(pi - 1, lnT, T);
        const gScalar lnkii  = lnki(pi, lnT, T);
        const gScalar f      = fp(log(p), pi);
        const gScalar k      = exp(lnkim1 + f * (lnkii - lnkim1));
        return k *
               (beta_[pi - 1] + f * (beta_[pi] - beta_[pi - 1]) +
                (Ta_[pi - 1] + f * (Ta_[pi] - Ta_[pi - 1])) / T) /
               T;
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
        return device_array<gScalar, 2>(
            {this->operator()(p, T, c), this->ddT(p, T, c)});
    }
};

} // namespace FoamGpu
