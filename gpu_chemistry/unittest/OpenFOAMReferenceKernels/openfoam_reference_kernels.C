#include "ODESolver.H"
#include "fundamentalConstants.H"
#include "openfoam_reference_kernels.H"
#include "physicoChemicalConstants.H"
#include "scalarMatrices.H"
#include "thermodynamicConstants.H"
#include "volFields.H"

#include "mock_of_odesystem.H"

namespace OFReferenceKernels {

TestData::constantResults constants() {
    TestData::constantResults ret;

    ret.RR     = Foam::constant::thermodynamic::RR;
    ret.pStd   = Foam::constant::thermodynamic::pStd;
    ret.Tstd   = Foam::constant::thermodynamic::Tstd;
    ret.NNA    = Foam::constant::physicoChemical::NNA.value();
    ret.k      = Foam::constant::physicoChemical::k.value();
    ret.vGreat = Foam::vGreat;
    ret.vSmall = Foam::vSmall;
    ret.small  = Foam::small;
    ret.great  = Foam::great;
    return ret;
}

TestData::perfectGasResult
perfect_gas(gScalar p, gScalar T, gScalar Y, gScalar molWeight) {

    const Foam::perfectGas<Foam::specie> eos(
        Foam::specie("temp", Y, molWeight));

    TestData::perfectGasResult ret;

    ret.R   = eos.R();
    ret.rho = eos.rho(p, T);
    ret.h   = eos.h(p, T);
    ret.Cp  = eos.Cp(p, T);
    ret.e   = eos.e(p, T);
    ret.Cv  = eos.Cv(p, T);
    ret.sp  = eos.sp(p, T);
    // ret.sv = eos.sv(p, T);
    ret.psi    = eos.psi(p, T);
    ret.Z      = eos.Z(p, T);
    ret.CpMCv  = eos.CpMCv(p, T);
    ret.alphav = eos.alphav(p, T);
    return ret;
}

TestData::thermoResults thermo_h(TestData::Mechanism mech) {
    const Foam::scalar p = TestData::pInf(mech);
    const Foam::scalar T = TestData::TInf(mech);

    auto thermos = TestData::makeCpuThermos_h(mech);

    const gLabel nThermo = thermos.size();

    TestData::thermoResults ret(nThermo);

    for (gLabel i = 0; i < thermos.size(); ++i) {
        ret.W[i]         = thermos[i].W();
        ret.Y[i]         = thermos[i].Y();
        ret.R[i]         = thermos[i].R();
        ret.Cp[i]        = thermos[i].Cp(p, T);
        ret.ha[i]        = thermos[i].ha(p, T);
        ret.hs[i]        = thermos[i].hs(p, T);
        ret.hf[i]        = thermos[i].hf();
        ret.s[i]         = thermos[i].s(p, T);
        ret.gStd[i]      = thermos[i].gStd(T);
        ret.dCpdT[i]     = thermos[i].dCpdT(p, T);
        ret.Cv[i]        = thermos[i].Cv(p, T);
        ret.es[i]        = thermos[i].es(p, T);
        ret.ea[i]        = thermos[i].ea(p, T);
        ret.K[i]         = thermos[i].K(p, T);
        ret.Kp[i]        = thermos[i].Kp(p, T);
        ret.Kc[i]        = thermos[i].Kc(p, T);
        ret.dKcdTbyKc[i] = thermos[i].dKcdTbyKc(p, T);
    }

    return ret;
}

TestData::thermoResults thermo_e(TestData::Mechanism mech) {
    const Foam::scalar p = TestData::pInf(mech);
    const Foam::scalar T = TestData::TInf(mech);

    auto thermos = TestData::makeCpuThermos_e(mech);

    const gLabel nThermo = thermos.size();

    TestData::thermoResults ret(nThermo);

    for (gLabel i = 0; i < thermos.size(); ++i) {
        ret.W[i]         = thermos[i].W();
        ret.Y[i]         = thermos[i].Y();
        ret.R[i]         = thermos[i].R();
        ret.Cp[i]        = thermos[i].Cp(p, T);
        ret.ha[i]        = thermos[i].ha(p, T);
        ret.hs[i]        = thermos[i].hs(p, T);
        ret.hf[i]        = thermos[i].hf();
        ret.s[i]         = thermos[i].s(p, T);
        ret.gStd[i]      = thermos[i].gStd(T);
        ret.dCpdT[i]     = thermos[i].dCpdT(p, T);
        ret.Cv[i]        = thermos[i].Cv(p, T);
        ret.es[i]        = thermos[i].es(p, T);
        ret.ea[i]        = thermos[i].ea(p, T);
        ret.K[i]         = thermos[i].K(p, T);
        ret.Kp[i]        = thermos[i].Kp(p, T);
        ret.Kc[i]        = thermos[i].Kc(p, T);
        ret.dKcdTbyKc[i] = thermos[i].dKcdTbyKc(p, T);
    }

    return ret;
}

TestData::reactionResults reaction(TestData::Mechanism mech) {
    const Foam::ReactionList<FoamGpu::testThermoType1> reactions(
        TestData::makeSpeciesTable(mech),
        TestData::makeCpuThermos_h(mech),
        TestData::makeReactionDict(mech));

    const gLabel nSpecie    = TestData::speciesCount(mech);
    const gLabel nEqns      = TestData::equationCount(mech);
    const gLabel nReactions = reactions.size();

    const Foam::scalarField c = [&]() {
        Foam::scalarField ret(nSpecie);
        // assign_test_concentration(ret, mech);
        fill_linear(ret);
        return ret;
    }();
    Foam::scalar p  = TestData::pInf(mech);
    Foam::scalar T  = TestData::TInf(mech);
    Foam::label  li = 0;

    // OpenFOAM-dev reaction Jacobian layout: [c_0..c_{n-1}, T]
    const gLabel       nEqnsRef = nSpecie + 1;
    Foam::List<gLabel> c2s;
    gLabel             Tsi  = nSpecie;
    Foam::scalarField  cTWork0(nEqnsRef, 0);
    Foam::scalarField  cTWork1(nEqnsRef, 0);

    Foam::reaction::state s(nSpecie);
    s.reset(p, T, c, li);

    std::vector<gScalar> Thigh(nReactions);
    std::vector<gScalar> Tlow(nReactions);
    std::vector<gScalar> Kc(nReactions);
    std::vector<gScalar> kf(nReactions);
    std::vector<gScalar> kr(nReactions);
    std::vector<gScalar> omega(nReactions);

    std::vector<std::vector<gScalar>> dNdtByV(nReactions);
    std::vector<std::vector<gScalar>> ddNdtByVdcTp(nReactions);

    for (Foam::label i = 0; i < reactions.size(); ++i) {
        // OpenFOAM-dev (9dfdc49d8e) applies a single pair of temperature
        // limits to the whole reaction scheme, held by the ReactionList.
        // Reaction<ThermoType>::Tlow/Thigh now resolve to the thermo's
        // (JANAF) limits, which are not the reaction limits.
        Thigh[i] = reactions.Thigh();
        Tlow[i]  = reactions.Tlow();
        Kc[i]    = reactions[i].Kc(p, T);
        kf[i]    = reactions[i].kf(s);
        kr[i]    = reactions[i].kr(s);

        // arbitrary
        Foam::scalar omegaf = 0.3;
        Foam::scalar omegar = 0.4;
        omega[i] = reactions[i].omega(s, omegaf, omegar);

        Foam::scalarField dNdtByV_f(c.size(), 0);
        reactions[i].dNdtByV(s,
                             dNdtByV_f,
                             false,
                             Foam::List<Foam::label>{});
        dNdtByV[i] =
            std::vector<gScalar>(dNdtByV_f.begin(), dNdtByV_f.end());

        // Reference is (nSpecie + 1)^2 in OpenFOAM-dev's [c, T] layout;
        // compare with the GPU result via drop_pressure_matrix()
        Foam::scalarSquareMatrix ddNdtByVdcT_f(nEqnsRef, 0);

        dNdtByV_f = 0; // probably not necessary
        reactions[i].ddNdtByVdcT(s,
                                 dNdtByV_f,
                                 ddNdtByVdcT_f,
                                 false,
                                 c2s,
                                 Tsi,
                                 cTWork0,
                                 cTWork1);

        ddNdtByVdcTp[i] = std::vector<gScalar>(
            ddNdtByVdcT_f.v(),
            ddNdtByVdcT_f.v() + ddNdtByVdcT_f.size());
    }
    TestData::reactionResults ret;
    ret.Thigh        = Thigh;
    ret.Tlow         = Tlow;
    ret.Kc           = Kc;
    ret.kf           = kf;
    ret.kr           = kr;
    ret.omega        = omega;
    ret.dNdtByV      = dNdtByV;
    ret.ddNdtByVdcTp = ddNdtByVdcTp;
    return ret;
}

std::tuple<std::vector<gScalar>,
           std::vector<gLabel>,
           std::vector<gScalar>>
lu(const std::vector<gScalar>& m_vals,
   const std::vector<gScalar>& s_vals) {
    gLabel size = std::sqrt(m_vals.size());

    Foam::scalarSquareMatrix matrix(size, 0);
    std::copy(m_vals.begin(), m_vals.end(), matrix.v());
    Foam::List<Foam::label> pivot(size, 0);
    Foam::scalarList        scale(size, Foam::scalar(0));
    Foam::scalarField       source(size);
    std::copy(s_vals.begin(), s_vals.end(), source.begin());

    gLabel sign;
    Foam::LUDecompose(matrix, pivot, scale, sign);
    Foam::LUBacksubstitute(matrix, pivot, source);

    auto m_ret =
        std::vector<gScalar>(matrix.v(), matrix.v() + size * size);
    auto p_ret = std::vector<gLabel>(pivot.begin(), pivot.end());
    auto s_ret = std::vector<gScalar>(source.begin(), source.end());

    return std::make_tuple(m_ret, p_ret, s_ret);
}

TestData::odeSystemResults odesystem(TestData::Mechanism mech) {
    const gLabel       nSpecie = TestData::speciesCount(mech);
    // OpenFOAM-dev layout [Y, T]; p is held by the system (pInf)
    const gLabel       nEqns   = TestData::equationCount(mech) - 1;
    Foam::MockOFSystem system(mech);
    system.setPressure(TestData::pInf(mech));

    TestData::odeSystemResults ret;

    const Foam::scalarField y0 = [=]() {
        // Fill in the GPU layout [Y, T, p] so that the Y values are
        // identical to those of the GPU test, then drop p
        Foam::scalarField y0_t(TestData::equationCount(mech));
        fill_linear(y0_t);
        y0_t[nSpecie]     = TestData::TInf(mech);
        y0_t.setSize(nSpecie + 1);

        // std::vector<gScalar> y0_vec =
        // TestData::get_solution_vector(mech);
        // std::copy(y0_vec.begin(), y0_vec.end(), y0_t.begin());
        return y0_t;
    }();

    const Foam::scalar time = 0.32423;

    const gLabel li = 0;

    {
        Foam::scalarField dy(nEqns, 0.31);
        system.derivatives(0.0, y0, li, dy);
        ret.derivative = std::vector<gScalar>(dy.begin(), dy.end());
    }

    {
        Foam::scalarField        dy(nEqns, 0.31);
        Foam::scalarSquareMatrix J(nEqns, 0.1);
        system.jacobian(time, y0, li, dy, J);
        ret.jacobian = std::vector<gScalar>(J.v(), J.v() + J.size());
    }

    return ret;
}

TestData::odeSystemResults odesystem(TestData::Mechanism         mech,
                                     const std::vector<gScalar>& y0) {
    const gLabel       nSpecie = TestData::speciesCount(mech);
    const gLabel       nEqns   = nSpecie + 1;
    Foam::MockOFSystem system(mech);
    system.setPressure(y0.back());

    Foam::scalarField y(nEqns);
    std::copy(y0.begin(), y0.begin() + nEqns, y.begin());

    TestData::odeSystemResults ret;
    const gLabel               li = 0;

    {
        Foam::scalarField dy(nEqns, 0.31);
        system.derivatives(0.0, y, li, dy);
        ret.derivative = std::vector<gScalar>(dy.begin(), dy.end());
    }

    {
        Foam::scalarField        dy(nEqns, 0.31);
        Foam::scalarSquareMatrix J(nEqns, 0.1);
        system.jacobian(0.0, y, li, dy, J);
        ret.jacobian = std::vector<gScalar>(J.v(), J.v() + J.size());
    }

    return ret;
}

std::vector<gScalar> ode_solve(TestData::Mechanism mech,
                               std::string         solver_name,
                               gScalar             xStart,
                               gScalar             xEnd,
                               gScalar             dxTry) {
    Foam::dictionary dict;
    dict.add("solver", solver_name);
    // gpuFoam's Rosenbrock solvers use scaled partial pivoting; OpenFOAM-dev
    // defaults to unpivoted LU with row filtering, so select pivoting to
    // compare like with like
    dict.add("pivot", true);
    Foam::MockOFSystem system(mech);

    auto ode = Foam::ODESolver::New(system, dict);

    Foam::scalarField y = [&]() {
        // GPU layout [Y, T, p] -> OpenFOAM-dev layout [Y, T] with p set on
        // the system
        std::vector<gScalar> v = TestData::get_solution_vector(mech);
        system.setPressure(v.back());
        Foam::scalarField    ret(v.size() - 1);
        std::copy(v.begin(), v.end() - 1, ret.begin());
        return ret;
    }();

    /*
    Foam::scalarField y = [=](){
        const gLabel nSpecie = TestData::speciesCount(mech);
        const gLabel nEqns = TestData::equationCount(mech);
        Foam::scalarField y0_t(nEqns);
        fill_linear(y0_t);
        y0_t[nSpecie] = TestData::TInf(mech);
        y0_t[nSpecie + 1] = TestData::pInf(mech);
        return y0_t;
    }();
    */
    const Foam::label li = 0;

    Foam::scalar dxTry_temp = dxTry;
    ode->solve(xStart, xEnd, y, li, dxTry_temp);

    return std::vector<gScalar>(y.begin(), y.end());
}

} // namespace OFReferenceKernels
