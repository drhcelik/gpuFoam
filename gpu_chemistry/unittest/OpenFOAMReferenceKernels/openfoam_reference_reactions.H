#pragma once

#include "create_foam_inputs.H"
#include "scalarMatrices.H"

namespace OFReferenceKernels {

//- Construct the OpenFOAM-dev reactions of the mechanism for the reference
//  kernels.
//
//  OpenFOAM-dev (62be3e2828) declares reactionRates::pLog::hasDdc = true
//  but pLog::ddc is NotImplemented, so the Jacobian of any pLog reaction
//  aborts. The pLog rate constant is a function of p and T only, so the pLog
//  reactions are constructed here with a pLog rate that has hasDdc = false,
//  as pLogReactionRate::hasDdc() before 62be3e2828. The rate constant and
//  its temperature derivative are those of OpenFOAM-dev pLog.
Foam::autoPtr<Foam::reactionList>
makeReactions(TestData::Mechanism                             m,
              const Foam::PtrList<FoamGpu::testThermoType1>& thermos);

//- The net reaction rate for each specie involved and its derivative w.r.t.
//  the concentration and temperature, as reaction::ddNdtByVdcT of
//  OpenFOAM-dev (62be3e2828) without reduction, with the derivative w.r.t.
//  each concentration added once.
//
//  For the reactants Jacobian density, reaction::ddNdtByVdcT adds the
//  derivative w.r.t. the concentration of a reactant once for every lhs
//  entry of the reactant, which counts a repeated reactant, e.g. the
//  irreversible reaction "CH2 + CH2 = H2 + C2H2", more than once.
void ddNdtByVdcT(const Foam::reaction&        r,
                 const Foam::reaction::state& s,
                 Foam::scalarField&           dNdtByV,
                 Foam::scalarSquareMatrix&    ddNdtByVdcT,
                 const Foam::label            Tj);

} // namespace OFReferenceKernels
