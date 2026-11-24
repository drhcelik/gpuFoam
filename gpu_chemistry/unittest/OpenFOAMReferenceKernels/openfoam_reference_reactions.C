#include "openfoam_reference_reactions.H"

#include "makeRateReaction.H"
#include "pLog_reactionRate.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {
namespace reactionRates {

//- OpenFOAM-dev pLog rate, independent of the concentrations
class pLogNoDdc : public pLog {
public:
    using pLog::pLog;

    //- The rate is not a function of concentration
    static const bool hasDdc = false;
};

} // namespace reactionRates

typedef reactions::Irreversible<reactionRates::pLogNoDdc>
    IrreversiblePLogNoDdc;

typedef reactions::Reversible<reactionRates::pLogNoDdc,
                              FoamGpu::testThermoType1>
    ReversiblePLogNoDdc;

typedef reactions::NonEquilibriumReversible<reactionRates::pLogNoDdc>
    NonEquilibriumReversiblePLogNoDdc;

template <>
const word IrreversiblePLogNoDdc::typeName =
    word(IrreversiblePLogNoDdc::typeName_()) + '<' +
    reactionRates::pLogNoDdc::type() + '>';

template <>
const word ReversiblePLogNoDdc::typeName =
    word(ReversiblePLogNoDdc::typeName_()) + '<' +
    reactionRates::pLogNoDdc::type() + ',' +
    FoamGpu::testThermoType1::typeName() + '>';

template <>
const word NonEquilibriumReversiblePLogNoDdc::typeName =
    word(NonEquilibriumReversiblePLogNoDdc::typeName_()) + '<' +
    reactionRates::pLogNoDdc::type() + '>';

} // namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace OFReferenceKernels {

Foam::autoPtr<Foam::reactionList>
makeReactions(TestData::Mechanism                             m,
              const Foam::PtrList<FoamGpu::testThermoType1>& thermos) {
    using namespace Foam;

    const speciesTable species = TestData::makeSpeciesTable(m);
    const dictionary   dict    = TestData::makeReactionDict(m);
    const SpecieThermosRef<FoamGpu::testThermoType1> specieThermos(
        thermos);

    autoPtr<reactionList> reactionsPtr(
        new reactionList(species, specieThermos, dict));
    reactionList& reactions = reactionsPtr();

    // Replace the pLog reactions, in the order of construction
    const dictionary& reactionsDict = dict.subDict("reactions");

    label i = 0;

    forAllConstIter(dictionary, reactionsDict, iter) {
        const dictionary& reactionDict =
            reactionsDict.subDict(iter().keyword());

        const word type = reactions[i].selectionType();

        if (type == "Irreversible<pLog>") {
            reactions.set(i,
                          new IrreversiblePLogNoDdc(
                              species, specieThermos, reactionDict));
        } else if (type == "Reversible<pLog>") {
            reactions.set(i,
                          new ReversiblePLogNoDdc(
                              species, specieThermos, reactionDict));
        } else if (type == "NonEquilibriumReversible<pLog>") {
            reactions.set(i,
                          new NonEquilibriumReversiblePLogNoDdc(
                              species, specieThermos, reactionDict));
        }

        ++i;
    }

    return reactionsPtr;
}

void ddNdtByVdcT(const Foam::reaction&        r,
                 const Foam::reaction::state& s,
                 Foam::scalarField&           dNdtByV,
                 Foam::scalarSquareMatrix&    ddNdtByVdcT,
                 const Foam::label            Tj) {
    using namespace Foam;

    // Net rate and derivatives. The derivatives w.r.t. the concentrations
    // not set by the reaction are zero.
    scalar      dOmegaDT;
    scalarField dOmegaDc(s.c.size(), Zero);
    const scalar omega = r.omega(s, dOmegaDT, dOmegaDc);

    // Rates
    r.dNdtByV(omega, dNdtByV, false, List<label>());

    // Derivatives w.r.t temperature
    forAll(r.lhs(), i) {
        ddNdtByVdcT(r.lhs()[i].index, Tj) -=
            r.lhs()[i].stoichCoeff * dOmegaDT;
    }
    forAll(r.rhs(), i) {
        ddNdtByVdcT(r.rhs()[i].index, Tj) +=
            r.rhs()[i].stoichCoeff * dOmegaDT;
    }

    // Derivatives w.r.t concentration
    forAll(s.c, j) {
        forAll(r.lhs(), i) {
            ddNdtByVdcT(r.lhs()[i].index, j) -=
                r.lhs()[i].stoichCoeff * dOmegaDc[j];
        }
        forAll(r.rhs(), i) {
            ddNdtByVdcT(r.rhs()[i].index, j) +=
                r.rhs()[i].stoichCoeff * dOmegaDc[j];
        }
    }
}

} // namespace OFReferenceKernels
