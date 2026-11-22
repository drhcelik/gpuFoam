#pragma once

#include "mechanisms.H"
#include "foam_thermo_types.H"

#include "speciesTable.H"

#include "reactionList.H"
#include "specieThermosRef.H"
#include "sensibleEnthalpy.H"

#include "sutherlandTransport.H"
#include "thermo.H"

//using FoamGpu::testThermoType1 = FoamGpu::testThermoType1;

namespace TestData {

Foam::dictionary makeThermoDict(TestData::Mechanism m);

Foam::speciesTable makeSpeciesTable(TestData::Mechanism m);

Foam::PtrList<FoamGpu::testThermoType1> makeCpuThermos_h(TestData::Mechanism m);
Foam::PtrList<FoamGpu::testThermoType3> makeCpuThermos_e(TestData::Mechanism m);

Foam::dictionary makeReactionDict(TestData::Mechanism m);

Foam::reactionList
makeCpuReactions(TestData::Mechanism m);
} // namespace TestData
