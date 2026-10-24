#pragma once

#include "gpuThermo.H"
#include "mechanisms.H"
#include "results.H"

namespace CpuTestKernels {

TestData::constantResults constants();

TestData::perfectGasResult
perfect_gas(gScalar p, gScalar T, gScalar Y, gScalar molWeight);

TestData::thermoResults thermo_h(TestData::Mechanism mech);
TestData::thermoResults thermo_e(TestData::Mechanism mech);
TestData::reactionResults reaction(TestData::Mechanism mech);

std::tuple<std::vector<gScalar>,
           std::vector<gLabel>,
           std::vector<gScalar>>
lu(const std::vector<gScalar>& m_vals,
   const std::vector<gScalar>& s_vals);

TestData::odeSystemResults odesystem(TestData::Mechanism mech);

//- ODE system derivative and Jacobian at the given state in the GPU
//  layout [Y, T, p]
TestData::odeSystemResults odesystem(TestData::Mechanism         mech,
                                     const std::vector<gScalar>& y0);

//- Integrate from xStart to xEnd, using the scaled partial pivoting LU
//  decomposition if pivot is true and otherwise the unpivoted LU
//  decomposition with row filtering (OpenFOAM-dev default)
std::vector<gScalar> ode_solve(TestData::Mechanism mech,
                               std::string         solver_name,
                               gScalar             xStart,
                               gScalar             xEnd,
                               gScalar             dxTry,
                               bool                pivot = true);

} // namespace CpuTestKernels