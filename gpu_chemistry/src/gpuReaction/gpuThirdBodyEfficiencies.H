#pragma once

#include "device_array.H"
#include "gpu_constants.H"

namespace FoamGpu {

class gpuThirdBodyEfficiencies {
public:
    using effArray = device_array<gScalar, 300>;
    using idxArray = device_array<gLabel, 300>;

    gLabel   nSpecie_{};
    gLabel   nNonUnity_{};
    effArray efficiencies_{};
    idxArray nonUnitySpecies_{};
    effArray nonUnityEfficienciesM1_{};

    gpuThirdBodyEfficiencies() = default;

    /// Construct from a full efficiency table (default 1). Builds the
    /// sparse non-unity lists used by M, matching OpenFOAM-dev
    /// 76dc82d590 (M = sum(c) + sum((eff-1)*c_i) for non-unity species).
    inline gpuThirdBodyEfficiencies(gLabel          nSpecie,
                                    const effArray& efficiencies)
        : nSpecie_(nSpecie)
        , nNonUnity_(0)
        , efficiencies_(efficiencies)
    {
        for (gLabel i = 0; i < nSpecie_; ++i)
        {
            if (efficiencies_[i] != gScalar(1))
            {
                nonUnitySpecies_[nNonUnity_] = i;
                nonUnityEfficienciesM1_[nNonUnity_] =
                    efficiencies_[i] - gScalar(1);
                ++nNonUnity_;
            }
        }
    }

    ///
    ///@brief Computes the concentration of the third-bodies.
    ///
    /// OpenFOAM-dev 76dc82d590: M = C + sum((eff-1)*c_i) over non-unity
    /// species, with C = sum(c). Numerically equal to sum(eff_i*c_i).
    ///
    ///@param c Concentration.
    ///@return gScalar The concentration of the third-bodies.
    ///
    inline CUDA_HOSTDEV gScalar
    M(const mdspan<const gScalar, 1> c) const {
        gScalar M = 0;
        for (gLabel i = 0; i < nSpecie_; ++i) {
            M += c[i];
        }
        for (gLabel i = 0; i < nNonUnity_; ++i) {
            M += nonUnityEfficienciesM1_[i] * c[nonUnitySpecies_[i]];
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
