#pragma once

#include "device_array.H"
#include "gpu_constants.H"

namespace FoamGpu {

class gpuThirdBodyEfficiencies {
public:
    using effArray = device_array<gScalar, 300>;
    using idxArray = device_array<gLabel, 300>;

    gLabel   nSpecie_{};
    gScalar  defaultEfficiency_{1};
    gLabel   nNonDefault_{};
    effArray efficiencies_{};
    idxArray nonDefaultSpecies_{};
    effArray nonDefaultEfficienciesMdefault_{};

    gpuThirdBodyEfficiencies() = default;

    /// Construct from a full efficiency table and the default efficiency.
    /// Builds the sparse lists of the species whose efficiency differs from
    /// the default, used by M, matching OpenFOAM-dev 852bc45364
    /// (M = defaultEfficiency*sum(c) + sum((eff-defaultEfficiency)*c_i)).
    inline gpuThirdBodyEfficiencies(gLabel          nSpecie,
                                    const effArray& efficiencies,
                                    gScalar defaultEfficiency = gScalar(1))
        : nSpecie_(nSpecie)
        , defaultEfficiency_(defaultEfficiency)
        , nNonDefault_(0)
        , efficiencies_(efficiencies)
    {
        for (gLabel i = 0; i < nSpecie_; ++i)
        {
            if (efficiencies_[i] != defaultEfficiency_)
            {
                nonDefaultSpecies_[nNonDefault_] = i;
                nonDefaultEfficienciesMdefault_[nNonDefault_] =
                    efficiencies_[i] - defaultEfficiency_;
                ++nNonDefault_;
            }
        }
    }

    ///
    ///@brief Computes the concentration of the third-bodies.
    ///
    /// OpenFOAM-dev 852bc45364: M = defaultEfficiency*C
    /// + sum((eff-defaultEfficiency)*c_i) over the species whose efficiency
    /// differs from the default, with C = sum(c). Numerically equal to
    /// sum(eff_i*c_i).
    ///
    ///@param c Concentration.
    ///@return gScalar The concentration of the third-bodies.
    ///
    inline CUDA_HOSTDEV gScalar
    M(const mdspan<const gScalar, 1> c) const {
        gScalar C = 0;
        for (gLabel i = 0; i < nSpecie_; ++i) {
            C += c[i];
        }
        gScalar M = defaultEfficiency_ * C;
        for (gLabel i = 0; i < nNonDefault_; ++i) {
            M += nonDefaultEfficienciesMdefault_[i] * c[nonDefaultSpecies_[i]];
        }
        return M;
    }

    ///
    ///@brief Calculates the derivative of M w.r.t concentrations.
    ///
    ///@param c Concentration.
    ///@return const effArray& The full efficiency table (dM/dc_i = eff_i).
    ///
    inline CUDA_HOSTDEV const effArray&
    dMdc(const mdspan<const gScalar, 1> c) const {
        (void)c;
        return efficiencies_;
    }
};

} // namespace FoamGpu
