#pragma once

#include "gpuThirdBodyEfficiencies.H"

namespace FoamGpu {

///
///@brief Chemically activated reaction rate, as OpenFOAM-dev
/// ChemicallyActivatedReactionRate without the activation function
/// Jacobian (CHEMICALLY_ACTIVATION_FUNCTION_JACOBIAN undefined).
///
template <class ReactionRate, class ChemicallyActivationFunction>
class gpuChemicallyActivatedReactionRate {

    ReactionRate k0_;

    ReactionRate kInf_;

    ChemicallyActivationFunction F_;

    gpuThirdBodyEfficiencies thirdBodyEfficiencies_;

public:
    gpuChemicallyActivatedReactionRate() = default;

    inline gpuChemicallyActivatedReactionRate(
        const ReactionRate&                 k0,
        const ReactionRate&                 kInf,
        const ChemicallyActivationFunction& F,
        const gpuThirdBodyEfficiencies&     tbes)
        : k0_(k0)
        , kInf_(kInf)
        , F_(F)
        , thirdBodyEfficiencies_(tbes) {}

    ///
    ///@brief Computes the the reaction rate.
    ///
    ///@param p Pressure.
    ///@param T Temperature.
    ///@param c Concentration.
    ///@return gScalar The reaction rate.
    ///
    inline CUDA_HOSTDEV gScalar
    operator()(const gScalar                  p,
               const gScalar                  T,
               const mdspan<const gScalar, 1> c) const {
        const gScalar k0   = k0_(p, T, c);
        const gScalar kInf = kInf_(p, T, c);
        const gScalar M    = thirdBodyEfficiencies_.M(c);
        const gScalar Pr   = k0 / kInf * M;
        const gScalar F    = F_(T, Pr);

        return k0 / (1.0 + Pr) * F;
    }

    ///
    ///@brief Computes the derivative of the reaction rate w.r.t
    /// temperature.
    ///
    ///@param p Pressure.
    ///@param T Temperature.
    ///@param c Concentration
    ///@return gScalar Derivative of the reaction rate w.r.t
    /// temperature.
    ///
    inline CUDA_HOSTDEV gScalar
    ddT(const gScalar                  p,
        const gScalar                  T,
        const mdspan<const gScalar, 1> c) const {
        const gScalar k0   = k0_(p, T, c);
        const gScalar kInf = kInf_(p, T, c);
        const gScalar M    = thirdBodyEfficiencies_.M(c);
        const gScalar Pr   = k0 / kInf * M;
        const gScalar F    = F_(T, Pr);

        const gScalar dk0dT = k0_.ddT(p, T, c);

        return dk0dT / (1.0 + Pr) * F;
    }

    ///
    ///@brief Determines if the rate is a function of concentration.
    ///
    ///@return True if the low-pressure rate is a function of
    /// concentration.
    ///
    inline CUDA_HOSTDEV bool hasDdc() const { return k0_.hasDdc(); }

    ///
    ///@brief Compute the derivative of the rate w.r.t concentration.
    ///
    ///@param p Pressure.
    ///@param T Temperature.
    ///@param c Concentration.
    ///@param ddc A span of at least size nSpcecie where the
    /// derivative is put.
    ///
    inline CUDA_HOSTDEV void ddc(const gScalar                  p,
                                 const gScalar                  T,
                                 const mdspan<const gScalar, 1> c,
                                 mdspan<gScalar, 1> ddc) const {

        const gScalar k0   = k0_(p, T, c);
        const gScalar kInf = kInf_(p, T, c);
        const gScalar M    = thirdBodyEfficiencies_.M(c);
        const gScalar Pr   = k0 / kInf * M;
        const gScalar F    = F_(T, Pr);

        k0_.ddc(p, T, c, ddc);

        for (gLabel i = 0; i < gLabel(ddc.size()); ++i) {
            ddc[i] *= 1.0 / (1.0 + Pr) * F;
        }
    }

    ///
    ///@brief Computes the reaction rate, the derivative of reaction
    /// rate w.r.t temperature and concentration.
    ///
    ///@param p Pressure.
    ///@param T Temperature.
    ///@param c Concentration.
    ///@param ddc A span of at least size nSpcecie where the
    /// derivative is put.
    ///@return device_array<gScalar, 2> A pair of scalars where the
    /// first value is the reaction rate and second value is the
    /// derivative w.r.t temperature.
    ///
    inline CUDA_HOSTDEV device_array<gScalar, 2>
                        everything(const gScalar                  p,
                                   const gScalar                  T,
                                   const mdspan<const gScalar, 1> c,
                                   mdspan<gScalar, 1>             ddc) const {

        const gScalar k0   = k0_(p, T, c);
        const gScalar kInf = kInf_(p, T, c);
        const gScalar M    = thirdBodyEfficiencies_.M(c);
        const gScalar Pr   = k0 / kInf * M;
        const gScalar F    = F_(T, Pr);

        const gScalar temp  = 1.0 / (1.0 + Pr) * F;
        const gScalar k     = k0 * temp;
        const gScalar dk0dT = k0_.ddT(p, T, c);
        const gScalar ddT   = dk0dT * temp;

        if (k0_.hasDdc()) {
            k0_.ddc(p, T, c, ddc);

            for (gLabel i = 0; i < gLabel(ddc.size()); ++i) {
                ddc[i] *= temp;
            }
        } else {
            for (gLabel i = 0; i < gLabel(ddc.size()); ++i) {
                ddc[i] = gScalar(0);
            }
        }

        return {k, ddT};
    }
};

} // namespace FoamGpu
