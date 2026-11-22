// This file defines any additional combinations of chemistryModel -
// thermo pairs that the user wishes to use and are not pre-compiled
// in OpenFOAM. Because of a bug in basicChemistryModel::New all pairs
// need to be pre-compiled. Currently only the ability to use
// coefficientWileMulticomponent mixture with logPolynomial transport
// is added. With standard chemistryModel this combination can be used
// through the dynamic code routines.

#include "makeChemistryModel.H"
#include "gpuChemistryModel.H"

// Thermo types
#include "makeFluidMulticomponentThermo.H"
#include "psiMulticomponentThermo.H"
#include "rhoFluidMulticomponentThermo.H"
#include "rhoThermo.H"

#include "coefficientMulticomponentMixture.H"
#include "coefficientWilkeMulticomponentMixture.H"
#include "singleComponentMixture.H"

// Reaction types
#include "Arrhenius_reactionRate.H"
#include "Janev_reactionRate.H"
#include "LandauTeller_reactionRate.H"
#include "makeRateReaction.H"
#include "powerSeries_reactionRate.H"
#include "ThirdBody_reactionRate.H"

#include "ChemicallyActivated_reactionRate.H"
#include "FallOff_reactionRate.H"

#include "Lindemann_fallOffFunction.H"
#include "SRI_fallOffFunction.H"
#include "Troe_fallOffFunction.H"

#include "forExtraCoeffGases.H"

namespace Foam {

forExtraCoeffGases(makeFluidMulticomponentThermos,
                   rhoFluidThermo,
                   rhoFluidMulticomponentThermo,
                   coefficientMulticomponentMixture);

forExtraCoeffGases(makeFluidMulticomponentThermos,
                   rhoFluidThermo,
                   rhoFluidMulticomponentThermo,
                   coefficientWilkeMulticomponentMixture);

forExtraCoeffGases(makeFluidMulticomponentThermos,
                   psiThermo,
                   psiMulticomponentThermo,
                   coefficientMulticomponentMixture);
forExtraCoeffGases(makeFluidMulticomponentThermos,
                   psiThermo,
                   psiMulticomponentThermo,
                   coefficientWilkeMulticomponentMixture);

// As OpenFOAM-dev 62be3e2828 the irreversible and non-equilibrium
// reversible reactions do not depend on the thermo and are instantiated in
// libspecie. Only the reversible reactions are instantiated for the
// additional thermo types.
namespace reactionRates {
typedef ThirdBody<Arrhenius> ThirdBodyArrhenius;

typedef FallOff<Arrhenius, fallOffFunctions::Lindemann>
    FallOffArrheniusLindemann;
typedef FallOff<Arrhenius, fallOffFunctions::Troe> FallOffArrheniusTroe;
typedef FallOff<Arrhenius, fallOffFunctions::SRI> FallOffArrheniusSRI;

typedef ChemicallyActivated<Arrhenius, fallOffFunctions::Lindemann>
    ChemicallyActivatedArrheniusLindemann;
typedef ChemicallyActivated<Arrhenius, fallOffFunctions::Troe>
    ChemicallyActivatedArrheniusTroe;
typedef ChemicallyActivated<Arrhenius, fallOffFunctions::SRI>
    ChemicallyActivatedArrheniusSRI;
} // namespace reactionRates

forExtraCoeffGases(makeReversibleReaction, dictionary, Arrhenius);
forExtraCoeffGases(makeReversibleReaction, dictionary, LandauTeller);
forExtraCoeffGases(makeReversibleReaction, dictionary, ThirdBodyArrhenius);
forExtraCoeffGases(makeReversibleReaction, dictionary, Janev);
forExtraCoeffGases(makeReversibleReaction, dictionary, powerSeries);

forExtraCoeffGases(makeReversibleReaction,
                   dictionary,
                   FallOffArrheniusLindemann);

forExtraCoeffGases(makeReversibleReaction,
                   dictionary,
                   FallOffArrheniusTroe);

forExtraCoeffGases(makeReversibleReaction,
                   dictionary,
                   FallOffArrheniusSRI);

forExtraCoeffGases(makeReversibleReaction,
                   dictionary,
                   ChemicallyActivatedArrheniusLindemann);

forExtraCoeffGases(makeReversibleReaction,
                   dictionary,
                   ChemicallyActivatedArrheniusTroe);

forExtraCoeffGases(makeReversibleReaction,
                   dictionary,
                   ChemicallyActivatedArrheniusSRI);

// Finally add the chemistry models
forExtraCoeffGases(makeChemistryModels, gpuChemistryModel);

} // namespace Foam
