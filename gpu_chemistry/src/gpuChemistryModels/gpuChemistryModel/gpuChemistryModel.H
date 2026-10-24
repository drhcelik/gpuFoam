#pragma once

#include <vector>

#include "chemistryModel.H"


#include "makeGpuReactions.H"
#include "makeGpuThermo.H"
#include "readGpuOdeInputs.H"

#include "DynamicField.H"
#include "UniformField.H"
#include "ODESystem.H"
#include "ReactionList.H"

#include "chemistryReductionMethod.H"

#include "gpuKernelEvaluator.H"
#include "multicomponentMixture.H"
#include "generatedCellZone.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {
namespace chemistryModels {

template<class cpuThermoType>
class gpuChemistryModel : public chemistryModel {

public:

private:
    /// @brief Class to call the reactions' pre- and post-evaluate functions
    /// around the host evaluation of the reactions, as
    /// OpenFOAM-dev chemistryModels::Standard::reactionEvaluationScope.
    class reactionEvaluationScope {
        const gpuChemistryModel<cpuThermoType>& chemistry_;

    public:
        reactionEvaluationScope(const gpuChemistryModel<cpuThermoType>& chemistry)
            : chemistry_(chemistry) {
            forAll(chemistry_.reactions_, i) {
                chemistry_.reactions_[i].preEvaluate();
            }
        }

        ~reactionEvaluationScope() {
            forAll(chemistry_.reactions_, i) {
                chemistry_.reactions_[i].postEvaluate();
            }
        }
    };

    //- Optional cellZone to limit where the chemistry is integrated
    generatedCellZone zone_;

    //- Reference to the multi component mixture
    const multicomponentMixture<cpuThermoType>& mixture_;

    //- Thermodynamic data of the species
    const PtrList<cpuThermoType>& specieThermos_;

    //- Reactions
    const ReactionList<cpuThermoType> reactions_;

    //- List of reaction rate per specie [kg/m^3/s]
    PtrList<volScalarField::Internal> RR_;

    //- The gpu kernel
    FoamGpu::GpuKernelEvaluator evaluator_;

private:
    /// @brief Get the values of the input field in the zone cells.
    /// @param f The field.
    /// @return Vector of the values of f in the zone cells.
    std::vector<gScalar> getZoneValues(const scalarField& f) const;

    /// @brief Get a vector of density values at old time in the zone cells.
    /// @return Vector of density values.
    std::vector<gScalar> getRho0() const;

    /// @brief Get a vector of pressure values at old time in the zone cells.
    /// @return Vector of pressure values.
    std::vector<gScalar> getP0() const;

    /// @brief Get a vector of temperature values at old time in the zone
    /// cells.
    /// @return Vector of temperature values.
    std::vector<gScalar> getT0() const;

    /// @brief Get a vector of chemistry time step values in the zone cells.
    /// @return Vector of chemistry time step values.
    std::vector<gScalar> getDeltaTChem() const;

    /// @brief Get a flat vector of size nZoneCells x (nSpecie + 1)
    /// corresponding to specie mass fraction values + temperature in each zone
    /// cell. The layout of the vector is [Y0_1, Y0_2 ...], where each Y0_i
    /// corresponds to the solution vector of zone cell i. The pressure is not integrated but held
    /// constant, see getP0().
    /// @return Flat vector mass fractions and temperature in each cell.
    std::vector<gScalar> getY0() const;

    /// @brief Builds the gpuKernel from the corresponding user inputs and cpu
    /// thermos and reactions.
    /// @param nEqns Number of equations.
    /// @param nSpecie  Number of species.
    /// @param reactions The reactions constructed on cpu memory.
    /// @param physicalProperties  The physicalProperties dictionary.
    /// @param chemistryProperties The chemistryProperties dictionary.
    /// @param mixture The mixture containing cpu thermos and compositions.
    /// @return A gpu kernel evaluator object which can be called to compute the
    /// reaction rate on gpu.
    static FoamGpu::GpuKernelEvaluator
    makeEvaluator(label                              nCells,
                  label                              nEqns,
                  label                              nSpecie,
                  const ReactionList<cpuThermoType>& reactions,
                  const dictionary&                  physicalProperties,
                  const dictionary&                  chemistryProperties,
                  const multicomponentMixture<cpuThermoType>& mixture);

    /// @brief Computes and updates the reaction rates (RR_) as well as
    /// chemistry time steps (deltaTChem_) of the zone cells.
    /// @param deltaT The current flow time step: a UniformField for a
    /// global time step or a scalarField for local time stepping.
    /// @return The minimum of the computed chemistry time steps.
    template<class DeltaTType>
    scalar computeRRAndChemDeltaT(const DeltaTType& deltaT);

public:
    //- Runtime type information
    TypeName("gpuChemistryModel");

    // Constructors

    /// @brief Construct a gpuChemistryModel based on the multicomponent thermo.
    /// @param thermo a fully constructed multicomponent thermo.
    gpuChemistryModel(const fluidMulticomponentThermo& thermo);

    gpuChemistryModel(const gpuChemistryModel&) = delete;
    virtual ~gpuChemistryModel() {}

    /// @brief Get the number of equations in the chemistry system (nSpecie +
    /// 1).
    /// @return Number of equations in the chemistry system.
    label nEqns() const;

    /// @brief Get the current cell count.
    /// @return Current cell count.
    label nCells() const;

    /// @brief Get the number of species in the chemistry system.
    /// @return Number of species in the chemistry system.
    label nSpecie() const override;

    /// @brief Get the number of reactions in the chemistry system.
    /// @return Number of reactions.
    label nReaction() const override;

    /// @brief Get the current reaction rates of the species [kg/m^3/s].
    /// @return The current reaction rates.
    const PtrList<volScalarField::Internal>& RR() const override;

    /// @brief Get the name of the reaction at index i.
    /// @param reactioni The reaction index to query the name for.
    /// @return The name of the reaction.
    virtual inline const word& reactionName
    (
        const label reactioni
    ) const override;

    /// @brief Get the reaction rate of reactioni
    /// @param reactioni The reaction index to query the specie reaction rates
    /// for.
    /// @return Reaction rate in reactioni [kg/m^3/s]
    tmp<volScalarField::Internal>
    reactionRR(const label reactioni) const override;

    /// @brief Get the reaction rates of the species involved in the reaction
    /// corresponding to the input reaction index reactioni.
    /// @param reactioni The reaction index to query the specie reaction rates
    /// for.
    /// @return Reaction rates of the species in reactioni [kg/m^3/s]
    PtrList<volScalarField::Internal> specieReactionRR
    (
        const label reactioni
    ) const override;

    /// @brief Calculates and updates the reaction rates (RR_) based on the
    /// current values of Y. Note that his runs on the cpu since no ode solve
    /// calls are required.
    void calculate() override;

    /// @brief Solve the reaction system for the input scalar (flow) time step.
    /// @param deltaT The current flow time step.
    /// @return The minimum chemistry time step.
    scalar solve(const scalar deltaT) override;

    /// @brief Solve the reaction system for the input scalarField (flow) time
    /// steps.
    /// @param deltaT The current cell-specific flow time steps.
    /// @return The minimum chemistry time step.
    scalar solve(const scalarField& deltaT) override;

    /// @brief Get the current cell-specific chemical time scales.
    /// @return Field of cell-specific chemical time scales.
    tmp<volScalarField::Internal> tc() const override;

    /// @brief Return the current heat release rates corresponding to the
    /// reaction rates (RR_).
    /// @return The heat release rate [kg/m/s^3].
    tmp<volScalarField::Internal> Qdot() const override;

    // Member Operators

    //- Disallow default bitwise assignment
    void operator=(const gpuChemistryModel&) = delete;
};

} // End namespace chemistryModels
} // End namespace Foam

#ifdef NoRepository
    #include "gpuChemistryModel.C"
#endif