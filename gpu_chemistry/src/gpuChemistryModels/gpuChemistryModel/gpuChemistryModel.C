#include "gpuChemistryModel.H"


namespace Foam {
namespace chemistryModels {

template<class cpuThermoType>
std::vector<gScalar>
gpuChemistryModel<cpuThermoType>::getZoneValues(const scalarField& f) const {
    const label          nZoneCells = zone_.nCells();
    std::vector<gScalar> ret(nZoneCells);
    for (label zci = 0; zci < nZoneCells; zci++) {
        ret[zci] = f[zone_.celli(zci)];
    }
    return ret;
}

template<class cpuThermoType>
std::vector<gScalar> gpuChemistryModel<cpuThermoType>::getRho0() const {
    const volScalarField& rho = this->mesh()
                                    .template lookupObject<volScalarField>(
                                        this->thermo().phasePropertyName("rho"))
                                    .oldTime();
    return getZoneValues(rho);
}

template<class cpuThermoType>
std::vector<gScalar> gpuChemistryModel<cpuThermoType>::getP0() const {
    return getZoneValues(this->thermo().p().oldTime());
}

template<class cpuThermoType>
std::vector<gScalar> gpuChemistryModel<cpuThermoType>::getT0() const {
    return getZoneValues(this->thermo().T().oldTime());
}

template<class cpuThermoType>
std::vector<gScalar> gpuChemistryModel<cpuThermoType>::getDeltaTChem() const {
    return getZoneValues(deltaTChem_);
}

template<class cpuThermoType>
std::vector<gScalar> gpuChemistryModel<cpuThermoType>::getY0() const {
    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    const volScalarField& T = this->thermo().T().oldTime();

    const label nZoneCells = zone_.nCells();

    std::vector<gScalar> Y0(nZoneCells * nEqns());

    auto s = make_mdspan(Y0, extents<2>{nZoneCells, nEqns()});

    for (label zci = 0; zci < nZoneCells; zci++) {
        const label celli = zone_.celli(zci);
        for (label i = 0; i < nSpecie(); ++i) {
            s(zci, i) = Yvf[i].oldTime()[celli];
        }
        s(zci, nSpecie()) = T[celli];
    }

    return Y0;
}

template<class cpuThermoType>
FoamGpu::GpuKernelEvaluator gpuChemistryModel<cpuThermoType>::makeEvaluator(
    label                                       nCells,
    label                                       nEqns,
    label                                       nSpecie,
    const ReactionList<cpuThermoType>&          reactions,
    const dictionary&                           physicalProperties,
    const dictionary&                           chemistryProperties,
    const multicomponentMixture<cpuThermoType>& mixture) {

    auto gpu_thermos =
        FoamGpu::makeGpuThermos(mixture.specieThermos(), physicalProperties);
    auto gpu_reactions = FoamGpu::makeGpuReactions(
        mixture.species(), chemistryProperties, gpu_thermos, reactions);
    scalar Treact = chemistryProperties.lookupOrDefault<scalar>("Treact", 0.0);
    
    Info << "    Treact: " << Treact << " K, (below which chemistry is not solved)" << endl;
    return FoamGpu::GpuKernelEvaluator(
        nCells,
        nEqns,
        nSpecie,
        gpu_thermos,
        gpu_reactions,
        FoamGpu::read_gpuODESolverInputs(
            chemistryProperties.typeDict("ode")),
        Treact);
}

template<class cpuThermoType>
gpuChemistryModel<cpuThermoType>::gpuChemistryModel(const fluidMulticomponentThermo& thermo)
    : chemistryModel(thermo)
    , zone_(this->mesh(), *this, true, true)
    , mixture_(dynamicCast<const multicomponentMixture<cpuThermoType>>(
          this->thermo()))
    , specieThermos_(mixture_.specieThermos())
    , reactions_(mixture_.species(), specieThermos_, this->mesh(), *this)
    , RR_(this->thermo().Y().size())
    , evaluator_(makeEvaluator(
          zone_.nCells(),
          nEqns(),
          nSpecie(),
          reactions_,
          this->mesh().lookupObject<dictionary>("physicalProperties"),
          *this,
          mixture_)) {

    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    // Create the fields for the chemistry sources
    forAll(RR_, fieldi) {
        RR_.set(fieldi,
                new volScalarField::Internal(
                    IOobject("RR." + Yvf[fieldi].name(),
                             this->mesh().time().name(),
                             this->mesh(),
                             IOobject::NO_READ,
                             IOobject::NO_WRITE),
                    thermo.mesh(),
                    dimensionedScalar(dimMass / dimVolume / dimTime, 0)));
    }
}

template<class cpuThermoType>
label gpuChemistryModel<cpuThermoType>::nEqns() const {
    // nEqns = number of species + temperature, the pressure is constant
    // during the integration as in OpenFOAM-dev (604bc31acc)
    return this->nSpecie() + 1;
}

template<class cpuThermoType>
label gpuChemistryModel<cpuThermoType>::nCells() const { return this->thermo().T().size(); }

template<class cpuThermoType>
label gpuChemistryModel<cpuThermoType>::nSpecie() const { return this->thermo().Y().size(); }

template<class cpuThermoType>
label gpuChemistryModel<cpuThermoType>::nReaction() const { return reactions_.size(); }

template<class cpuThermoType>
const word& gpuChemistryModel<cpuThermoType>::reactionName
(
    const label reactioni
) const { 
        return reactions_[reactioni].name(); 
}


template<class cpuThermoType>
const PtrList<volScalarField::Internal>& gpuChemistryModel<cpuThermoType>::RR() const {
    return RR_;
}

template<class cpuThermoType>
void gpuChemistryModel<cpuThermoType>::calculate() {
    if (!this->chemistry_) { return; }

    if (!zone_.all()) {
        forAll(RR_, fieldi) { RR_[fieldi] = Zero; }
    }

    tmp<volScalarField>   trhovf(this->thermo().rho());
    const volScalarField& rhovf = trhovf();

    const volScalarField& Tvf = this->thermo().T();
    const volScalarField& pvf = this->thermo().p();

    scalarField dNdtByV(this->nSpecie() + 1);

    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    reaction::state rs(this->nSpecie());
    scalarField&    c = rs.c;

    List<label> cTos; // Note! not allocated!

    reactionEvaluationScope scope(*this);

    const label nZoneCells = zone_.nCells();
    for (label zci = 0; zci < nZoneCells; zci++) {
        const label celli = zone_.celli(zci);
        const scalar rho = rhovf[celli];
        const scalar T   = Tvf[celli];
        const scalar p   = pvf[celli];

        for (label i = 0; i < this->nSpecie(); i++) {
            const scalar Yi = Yvf[i][celli];
            c[i]            = rho * Yi / specieThermos_[i].W();
        }

        rs.reset(p, T, celli);

        dNdtByV = Zero;

        forAll(reactions_, ri) {
            reactions_[ri].dNdtByV(rs, dNdtByV, false, cTos);
        }

        for (label i = 0; i < this->nSpecie(); i++) {
            RR_[i][celli] = dNdtByV[i] * specieThermos_[i].W();
        }
    }
}

template<class cpuThermoType>
tmp<volScalarField::Internal>
gpuChemistryModel<cpuThermoType>::reactionRR(const label reactioni) const {
    
    tmp<volScalarField::Internal> tRR =
        volScalarField::Internal::New
        (
            "RR:" + reactions_[reactioni].name(),
            this->mesh(),
            dimensionedScalar(dimMoles/dimVolume/dimTime, 0)
        );
    volScalarField::Internal& RR = tRR.ref();

    if (!this->chemistry_) { return tRR; }


    tmp<volScalarField>   trhovf(this->thermo().rho());
    const volScalarField& rhovf = trhovf();

    const volScalarField& Tvf = this->thermo().T();
    const volScalarField& pvf = this->thermo().p();
    
    const auto& R = reactions_[reactioni];

    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    reaction::state rs(this->nSpecie());
    scalarField&    c = rs.c;

    reactionEvaluationScope scope(*this);

    const label nZoneCells = zone_.nCells();
    for (label zci = 0; zci < nZoneCells; zci++) {
        const label celli = zone_.celli(zci);
        const scalar rho = rhovf[celli];
        const scalar T = Tvf[celli];
        const scalar p = pvf[celli];

        for (label i=0; i<this->nSpecie(); i++)
        {
            const scalar Yi = Yvf[i][celli];
            c[i] = rho*Yi/specieThermos_[i].W();
        }

        rs.reset(p, T, celli);

        scalar omegaf, omegar;

        RR[celli] =
            R.omega
            (
                rs,
                omegaf,
                omegar
            );
    }
    return tRR;

}

template<class cpuThermoType>
PtrList<volScalarField::Internal>
gpuChemistryModel<cpuThermoType>::specieReactionRR(const label reactioni) const {
    

    PtrList<volScalarField::Internal> RR(this->nSpecie());
    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    for (label i=0; i<this->nSpecie(); i++)
    {
        RR.set
        (
            i,
            volScalarField::Internal::New
            (
                "RR:" + reactions_[reactioni].name() + ":" + Yvf[i].name(),
                this->mesh(),
                dimensionedScalar(dimMass/dimVolume/dimTime, 0)
            ).ptr()
        );
    }

    if (!this->chemistry_)
    {
        return RR;
    }

    tmp<volScalarField> trhovf(this->thermo().rho());
    const volScalarField& rhovf = trhovf();

    const volScalarField& Tvf = this->thermo().T();
    const volScalarField& pvf = this->thermo().p();

    scalarField dNdtByV(this->nSpecie() + 1);


    const auto& R = reactions_[reactioni];

    List<label> cTos; // Note! not allocated!
    reaction::state rs(this->nSpecie());
    scalarField&    c = rs.c;


    reactionEvaluationScope scope(*this);

    const label nZoneCells = zone_.nCells();
    for (label zci = 0; zci < nZoneCells; zci++) {
        const label celli = zone_.celli(zci);
        const scalar rho = rhovf[celli];
        const scalar T = Tvf[celli];
        const scalar p = pvf[celli];

        for (label i=0; i<this->nSpecie(); i++)
        {
            const scalar Yi = Yvf[i][celli];
            c[i] = rho*Yi/specieThermos_[i].W();
        }

        rs.reset(p, T, celli);

        dNdtByV = Zero;

        R.dNdtByV
        (
            rs,
            dNdtByV,
            false,
            cTos
        );

        for (label i=0; i<this->nSpecie(); i++)
        {
            RR[i][celli] = dNdtByV[i]*specieThermos_[i].W();
        }
    }

    return RR;
}

template<class cpuThermoType>
scalar gpuChemistryModel<cpuThermoType>::solve(const scalar deltaT) {
    // Don't allow the time-step to change more than a factor of 2
    return min(
        this->computeRRAndChemDeltaT(UniformField<scalar>(deltaT)),
        2 * deltaT);
}

template<class cpuThermoType>
scalar gpuChemistryModel<cpuThermoType>::solve(const scalarField& deltaT) {
    // Local time stepping, as OpenFOAM-dev chemistryModels::Standard
    return this->computeRRAndChemDeltaT(deltaT);
}

template<class cpuThermoType>
template<class DeltaTType>
scalar gpuChemistryModel<cpuThermoType>::computeRRAndChemDeltaT(
    const DeltaTType& deltaT) {

    if (!this->chemistry_) { return great; }

    // As OpenFOAM-dev chemistryModels::Standard (d5522d021b) the reaction
    // rates are zero outside the zone and the chemistry time step of the
    // cells outside the zone is not changed
    if (!zone_.all()) {
        forAll(RR_, fieldi) { RR_[fieldi] = Zero; }
    }

    zone_.regenerate();
    const label nZoneCells = zone_.nCells();

    if (nZoneCells == 0) { return great; }

    // The flow time step of the zone cells
    std::vector<gScalar> zoneDeltaT(nZoneCells);
    for (label zci = 0; zci < nZoneCells; zci++) {
        zoneDeltaT[zci] = deltaT[zone_.celli(zci)];
    }

    auto tpl = evaluator_.computeRR(zoneDeltaT,
                                    deltaTChemMax_,
                                    getRho0(),
                                    getDeltaTChem(),
                                    getP0(),
                                    getY0());

    auto RR = std::get<0>(tpl);
    auto deltaTChem = std::get<1>(tpl);
    auto minDeltaT = std::get<2>(tpl);

    auto RRs = make_mdspan(RR, extents<2>{nZoneCells, nSpecie()});

    for (label zci = 0; zci < nZoneCells; zci++) {
        const label celli = zone_.celli(zci);
        for (label i = 0; i < this->nSpecie(); ++i) {
            RR_[i][celli] = RRs(zci, i);
        }
        deltaTChem_[celli] = deltaTChem[zci];
    }

    return minDeltaT;
}

template<class cpuThermoType>
tmp<volScalarField::Internal>
gpuChemistryModel<cpuThermoType>::tc() const {
    tmp<volScalarField::Internal> ttc(volScalarField::Internal::New(
        "tc",
        this->mesh(),
        dimensionedScalar(dimensions::time, small)));
    scalarField& tc = ttc.ref();

    if (!this->chemistry_) { return ttc; }

    tmp<volScalarField>   trhovf(this->thermo().rho());
    const volScalarField& rhovf = trhovf();

    const volScalarField& Tvf = this->thermo().T();
    const volScalarField& pvf = this->thermo().p();

    scalarField dNdtByV(this->nSpecie() + 1);

    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    reaction::state rs(this->nSpecie());
    scalarField&    c = rs.c;

    List<label> cTos; // Note! not allocated!

    reactionEvaluationScope scope(*this);

    const label nZoneCells = zone_.nCells();
    for (label zci = 0; zci < nZoneCells; zci++) {
        const label celli = zone_.celli(zci);
        const scalar rho = rhovf[celli];
        const scalar T   = Tvf[celli];
        const scalar p   = pvf[celli];

        for (label i = 0; i < this->nSpecie(); i++) {
            c[i] = rho * Yvf[i][celli] / specieThermos_[i].W();
        }

        rs.reset(p, T, celli);

        // A reaction's rate scale is calculated as it's molar
        // production rate divided by the total number of moles in the
        // system.
        //
        // The system rate scale is the average of the reactions' rate
        // scales weighted by the reactions' molar production rates. This
        // weighting ensures that dominant reactions provide the largest
        // contribution to the system rate scale.
        //
        // The system time scale is then the reciprocal of the system rate
        // scale.
        //
        // Contributions from forward and reverse reaction rates are
        // handled independently and identically so that reversible
        // reactions produce the same result as the equivalent pair of
        // irreversible reactions.

        scalar sumW = 0, sumWRateByCTot = 0;
        forAll(reactions_, i) {
            const Reaction<cpuThermoType>& R = reactions_[i];
            scalar                         omegaf, omegar;
            R.omega(rs, omegaf, omegar);

            scalar wf = 0;
            forAll(R.rhs(), j) { wf += R.rhs()[j].stoichCoeff * omegaf; }
            sumW += wf;
            sumWRateByCTot += sqr(wf);

            scalar wr = 0;
            forAll(R.lhs(), j) { wr += R.lhs()[j].stoichCoeff * omegar; }
            sumW += wr;
            sumWRateByCTot += sqr(wr);
        }

        tc[celli] =
            sumWRateByCTot == 0 ? vGreat : sumW / sumWRateByCTot * sum(c);
    }

    return ttc;
}

template<class cpuThermoType>
tmp<volScalarField::Internal>
gpuChemistryModel<cpuThermoType>::Qdot() const {
    tmp<volScalarField::Internal> tQdot(volScalarField::Internal::New(
        "Qdot",
        this->mesh_,
        dimensionedScalar(dimensions::powerDensity, 0)));

    if (!this->chemistry_) { return tQdot; }

    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    scalarField& Qdot = tQdot.ref();

    reactionEvaluationScope scope(*this);

    const label nZoneCells = zone_.nCells();

    forAll(Yvf, i) {
        for (label zci = 0; zci < nZoneCells; zci++) {
            const label celli = zone_.celli(zci);
            const scalar hi = specieThermos_[i].hf();
            Qdot[celli] -= hi * RR_[i][celli];
        }
    }

    return tQdot;
}

} // namespace chemistryModels
} // namespace Foam
