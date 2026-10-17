#include "catch.H"
#include <cmath>

#include "openfoam_reference_kernels.H"
#include "gpu_test_kernels.H"
#include "cpu_test_kernels.H"
#include "test_utilities.H"
#include "mechanisms.H"
#include "mdspan.H"
#include "gpuSpecieExponent.H"
#include "makeGpuReactionRate.H"
#include "reaction.H"
#include "thirdBodyEfficiencies.H"
#include "IStringStream.H"


TEST_CASE("make_mdspan", "[CPU]"){

    std::vector<int> v = {1,2,3,4,5,6};
    auto s = make_mdspan(v, extents<2>(2,3));
    CHECK(s(0,0) == 1);

    CHECK(s.size() == 2*3);

}



TEST_CASE("Test gpuConstants (on CPU)", "[CPU]")
{
    auto reference = OFReferenceKernels::constants();
    auto test_result = CpuTestKernels::constants();

    SECTION("Physical")
    {
        CHECK(test_result.RR == reference.RR);
        CHECK(test_result.pStd == reference.pStd);
        CHECK(test_result.Tstd == reference.Tstd);
        CHECK(test_result.NNA == reference.NNA);
        CHECK(test_result.k == reference.k);

    }

    SECTION("Numeric")
    {
        CHECK(test_result.vGreat == reference.vGreat);
        CHECK(test_result.vSmall == reference.vSmall);
        CHECK(test_result.small == reference.small);
        CHECK(test_result.great == reference.great);

    }
}

TEST_CASE("Test gpuConstants (on GPU)", "[GPU]")
{
    auto reference = OFReferenceKernels::constants();
    auto test_result = GpuTestKernels::constants();

    SECTION("Physical")
    {
        CHECK(test_result.RR == reference.RR);
        CHECK(test_result.pStd == reference.pStd);
        CHECK(test_result.Tstd == reference.Tstd);
        CHECK(test_result.NNA == reference.NNA);
        CHECK(test_result.k == reference.k);

    }

    SECTION("Numeric")
    {
        CHECK(test_result.vGreat == reference.vGreat);
        CHECK(test_result.vSmall == reference.vSmall);
        CHECK(test_result.small == reference.small);
        CHECK(test_result.great == reference.great);

    }
}



TEST_CASE("Test perfectGas (on CPU)", "[CPU]"){

    using namespace FoamGpu;

    SECTION("thermo properties")
    {
        const gScalar p = 1E5;
        const gScalar T = 3542.324;
        const gScalar molWeight = 0.32;
        const gScalar Y = 0.1;

        auto reference = OFReferenceKernels::perfect_gas(p, T, Y, molWeight);
        auto test_result = CpuTestKernels::perfect_gas(p, T, Y, molWeight);

        CHECK(test_result.R == reference.R);
        CHECK(test_result.rho == reference.rho);
        CHECK(test_result.h == reference.h);
        CHECK(test_result.Cp == reference.Cp);
        CHECK(test_result.e == reference.e);
        CHECK(test_result.Cv == reference.Cv);
        CHECK(test_result.sp == reference.sp);
        CHECK(test_result.psi == reference.psi);
        CHECK(test_result.Z == reference.Z);
        CHECK(test_result.CpMCv == reference.CpMCv);
        CHECK(test_result.alphav == reference.alphav);

    }

}

TEST_CASE("Test perfectGas (on GPU)", "[GPU]"){

    using namespace FoamGpu;

    SECTION("thermo properties")
    {
        const gScalar p = 1E5;
        const gScalar T = 3542.324;
        const gScalar molWeight = 0.32;
        const gScalar Y = 0.1;

        auto reference = OFReferenceKernels::perfect_gas(p, T, Y, molWeight);
        auto test_result = GpuTestKernels::perfect_gas(p, T, Y, molWeight);

        CHECK(test_result.R == reference.R);
        CHECK(test_result.rho == reference.rho);
        CHECK(test_result.h == reference.h);
        CHECK(test_result.Cp == reference.Cp);
        CHECK(test_result.e == reference.e);
        CHECK(test_result.Cv == reference.Cv);
        CHECK(test_result.sp == reference.sp);
        CHECK(test_result.psi == reference.psi);
        CHECK(test_result.Z == reference.Z);
        CHECK(test_result.CpMCv == reference.CpMCv);
        CHECK(test_result.alphav == reference.alphav);

    }

}




static inline void thermoTests(TestData::thermoResults& test_result, TestData::thermoResults& reference, double errorTol)
{

    CHECK_THAT
    (
        reference.W,
        Catch::Matchers::Approx(test_result.W).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.Y,
        Catch::Matchers::Approx(test_result.Y).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.R,
        Catch::Matchers::Approx(test_result.R).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.Cp,
        Catch::Matchers::Approx(test_result.Cp).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.ha,
        Catch::Matchers::Approx(test_result.ha).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.hs,
        Catch::Matchers::Approx(test_result.hs).epsilon(errorTol)
    );


    remove_negative(reference.hf, errorTol);
    remove_negative(test_result.hf, errorTol);

    CHECK_THAT
    (
        reference.hf,
        Catch::Matchers::Approx(test_result.hf).epsilon(errorTol)
    );

    CHECK_THAT
    (
        reference.s,
        Catch::Matchers::Approx(test_result.s).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.gStd,
        Catch::Matchers::Approx(test_result.gStd).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.dCpdT,
        Catch::Matchers::Approx(test_result.dCpdT).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.Cv,
        Catch::Matchers::Approx(test_result.Cv).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.es,
        Catch::Matchers::Approx(test_result.es).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.ea,
        Catch::Matchers::Approx(test_result.ea).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.K,
        Catch::Matchers::Approx(test_result.K).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.Kp,
        Catch::Matchers::Approx(test_result.Kp).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.Kc,
        Catch::Matchers::Approx(test_result.Kc).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.dKcdTbyKc,
        Catch::Matchers::Approx(test_result.dKcdTbyKc).epsilon(errorTol)
    );


}

TEST_CASE("Test gpuThermo h (on CPU)", "[CPU]")
{

    constexpr double errorTol = 1E-9;
    SECTION("GRI")
    {
        auto test_result = CpuTestKernels::thermo_h(TestData::GRI);
        auto reference = OFReferenceKernels::thermo_h(TestData::GRI);
        thermoTests(test_result, reference, errorTol);
    }

    /*
    SECTION("YAO")
    {
        auto test_result = CpuTestKernels::thermo_h(TestData::YAO);
        auto reference = OFReferenceKernels::thermo_h(TestData::YAO);
        thermoTests(test_result, reference, errorTol);
    }
    */

    SECTION("H2")
    {
        auto test_result = CpuTestKernels::thermo_h(TestData::H2);
        auto reference = OFReferenceKernels::thermo_h(TestData::H2);
        thermoTests(test_result, reference, errorTol);
    }

}

TEST_CASE("Test gpuThermo h (on GPU)", "[GPU]")
{

    constexpr double errorTol = 1E-7;
    SECTION("GRI")
    {
        auto test_result = GpuTestKernels::thermo_h(TestData::GRI);
        auto reference = OFReferenceKernels::thermo_h(TestData::GRI);
        thermoTests(test_result, reference, errorTol);
    }

    SECTION("YAO")
    {
        auto test_result = GpuTestKernels::thermo_h(TestData::YAO);
        auto reference = OFReferenceKernels::thermo_h(TestData::YAO);
        thermoTests(test_result, reference, errorTol);
    }

    SECTION("H2")
    {
        auto test_result = GpuTestKernels::thermo_h(TestData::H2);
        auto reference = OFReferenceKernels::thermo_h(TestData::H2);
        thermoTests(test_result, reference, errorTol);
    }

}

TEST_CASE("Test gpuThermo e (on CPU)", "[CPU]")
{

    constexpr double errorTol = 1E-9;
    SECTION("GRI")
    {
        auto test_result = CpuTestKernels::thermo_e(TestData::GRI);
        auto reference = OFReferenceKernels::thermo_e(TestData::GRI);
        thermoTests(test_result, reference, errorTol);
    }

    SECTION("YAO")
    {
        auto test_result = CpuTestKernels::thermo_e(TestData::YAO);
        auto reference = OFReferenceKernels::thermo_e(TestData::YAO);
        thermoTests(test_result, reference, errorTol);
    }

    SECTION("H2")
    {
        auto test_result = CpuTestKernels::thermo_e(TestData::H2);
        auto reference = OFReferenceKernels::thermo_e(TestData::H2);
        thermoTests(test_result, reference, errorTol);
    }

}

TEST_CASE("Test gpuThermo e (on GPU)", "[GPU]")
{

    constexpr double errorTol = 1E-7;
    SECTION("GRI")
    {
        auto test_result = GpuTestKernels::thermo_e(TestData::GRI);
        auto reference = OFReferenceKernels::thermo_e(TestData::GRI);
        thermoTests(test_result, reference, errorTol);
    }

    SECTION("YAO")
    {
        auto test_result = GpuTestKernels::thermo_e(TestData::YAO);
        auto reference = OFReferenceKernels::thermo_e(TestData::YAO);
        thermoTests(test_result, reference, errorTol);
    }


    SECTION("H2")
    {
        auto test_result = GpuTestKernels::thermo_e(TestData::H2);
        auto reference = OFReferenceKernels::thermo_e(TestData::H2);
        thermoTests(test_result, reference, errorTol);
    }

}


static inline void reactionTests(const TestData::reactionResults& test_result, const TestData::reactionResults& reference, double errorTol)
{

    using namespace FoamGpu;


    CHECK_THAT
    (
        reference.Thigh,
        Catch::Matchers::Approx(test_result.Thigh).epsilon(errorTol)
    );

    CHECK_THAT
    (
        reference.Tlow,
        Catch::Matchers::Approx(test_result.Tlow).epsilon(errorTol)
    );

    CHECK_THAT
    (
        reference.Kc,
        Catch::Matchers::Approx(test_result.Kc).epsilon(errorTol)
    );

    CHECK_THAT
    (
        reference.kf,
        Catch::Matchers::Approx(test_result.kf).epsilon(errorTol)
    );

    CHECK_THAT
    (
        reference.kr,
        Catch::Matchers::Approx(test_result.kr).epsilon(errorTol)
    );

    CHECK_THAT
    (
        reference.omega,
        Catch::Matchers::Approx(test_result.omega).epsilon(errorTol)
    );


    for (size_t i = 0; i < reference.dNdtByV.size(); ++i)
    {
        REQUIRE_THAT
        (
            reference.dNdtByV[i],
            Catch::Matchers::Approx(test_result.dNdtByV[i]).epsilon(errorTol)
        );
    }


    for (size_t i = 0; i < reference.ddNdtByVdcTp.size(); ++i)
    {
        REQUIRE_THAT
        (
            reference.ddNdtByVdcTp[i],
            Catch::Matchers::Approx(drop_pressure_matrix(test_result.ddNdtByVdcTp[i])).epsilon(errorTol)
        );
    }


}


TEST_CASE("Test gpuReaction (on CPU)", "[CPU]")
{
    constexpr double errorTol = 1E-9;

    SECTION("GRI")
    {

        auto test_result = CpuTestKernels::reaction(TestData::GRI);
        auto reference = OFReferenceKernels::reaction(TestData::GRI);
        reactionTests(test_result, reference, errorTol);
    }

    SECTION("YAO")
    {

        auto test_result = CpuTestKernels::reaction(TestData::YAO);
        auto reference = OFReferenceKernels::reaction(TestData::YAO);
        reactionTests(test_result, reference, errorTol);
    }



    SECTION("H2")
    {

        auto test_result = CpuTestKernels::reaction(TestData::H2);
        auto reference = OFReferenceKernels::reaction(TestData::H2);
        reactionTests(test_result, reference, errorTol);
    }
}


TEST_CASE("Test gpuReaction (on GPU)", "[GPU]")
{
    constexpr double errorTol = 1E-9;


    SECTION("GRI")
    {

        auto test_result = GpuTestKernels::reaction(TestData::GRI);
        auto reference = OFReferenceKernels::reaction(TestData::GRI);
        reactionTests(test_result, reference, errorTol);
    }

    SECTION("YAO")
    {

        auto test_result = GpuTestKernels::reaction(TestData::YAO);
        auto reference = OFReferenceKernels::reaction(TestData::YAO);
        reactionTests(test_result, reference, errorTol);
    }

    SECTION("H2")
    {

        auto test_result = GpuTestKernels::reaction(TestData::H2);
        auto reference = OFReferenceKernels::reaction(TestData::H2);
        reactionTests(test_result, reference, errorTol);
    }



}



TEST_CASE("Test ludecompose (on CPU)", "[CPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-9;


    for (int i = 3; i < 50; ++i)
    {
        int size = i;


        std::vector<gScalar> vals(size*size);
        fill_random(vals);
        std::vector<gScalar> source(size, 1);


        auto [m_test, p_test, s_test] = CpuTestKernels::lu(vals, source);
        auto [m_ref, p_ref, s_ref] = OFReferenceKernels::lu(vals, source);

        // The LU factors and pivot indices are not compared: OpenFOAM-dev's
        // ikj LUDecompose (10f026fd67) breaks ties in the scaled pivot
        // search by taking the first candidate row, whereas gpuFoam's Crout
        // form takes the last.  Ties are common (every row whose largest
        // coefficient lies in the pivot column has ratio 1), so both give
        // valid but different factorisations.  The solution is unique.
        (void)m_test; (void)m_ref; (void)p_test; (void)p_ref;
        REQUIRE_THAT
        (
            s_test,
            // Absolute margin: the two pivot sequences give round-off
            // differences of O(cond(A)*eps*|x|), |x| ~ 1, which a purely
            // relative check rejects for near-zero solution components
            Catch::Matchers::Approx(s_ref).epsilon(errorTol).margin(1e-12)
        );

    }
}


TEST_CASE("Test ludecompose (on GPU)", "[GPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-9;


    for (int i = 3; i < 50; ++i)
    {
        int size = i;

        std::vector<gScalar> vals(size*size);
        fill_random(vals);
        std::vector<gScalar> source(size, 1);


        auto [m_test, p_test, s_test] = GpuTestKernels::lu(vals, source);
        auto [m_ref, p_ref, s_ref] = OFReferenceKernels::lu(vals, source);

        // The LU factors and pivot indices are not compared: OpenFOAM-dev's
        // ikj LUDecompose (10f026fd67) breaks ties in the scaled pivot
        // search by taking the first candidate row, whereas gpuFoam's Crout
        // form takes the last.  Ties are common (every row whose largest
        // coefficient lies in the pivot column has ratio 1), so both give
        // valid but different factorisations.  The solution is unique.
        (void)m_test; (void)m_ref; (void)p_test; (void)p_ref;
        REQUIRE_THAT
        (
            s_test,
            // Absolute margin: the two pivot sequences give round-off
            // differences of O(cond(A)*eps*|x|), |x| ~ 1, which a purely
            // relative check rejects for near-zero solution components
            Catch::Matchers::Approx(s_ref).epsilon(errorTol).margin(1e-12)
        );

    }
}




TEST_CASE("Test gpuOdeSystem (on CPU)", "[CPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-9;


    SECTION("H2"){

        auto test_result = CpuTestKernels::odesystem(TestData::Mechanism::H2);
        auto reference = OFReferenceKernels::odesystem(TestData::Mechanism::H2);


        CHECK_THAT
        (
            drop_pressure(test_result.derivative),
            Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
        );

        CHECK_THAT
        (
            drop_pressure_matrix(test_result.jacobian),
            Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
        );
    }
    SECTION("YAO"){

        auto test_result = CpuTestKernels::odesystem(TestData::Mechanism::YAO);
        auto reference = OFReferenceKernels::odesystem(TestData::Mechanism::YAO);


        CHECK_THAT
        (
            drop_pressure(test_result.derivative),
            Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
        );

        CHECK_THAT
        (
            drop_pressure_matrix(test_result.jacobian),
            Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
        );
    }
    SECTION("GRI"){

        auto test_result = CpuTestKernels::odesystem(TestData::Mechanism::GRI);
        auto reference = OFReferenceKernels::odesystem(TestData::Mechanism::GRI);

        CHECK_THAT
        (
            drop_pressure(test_result.derivative),
            Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
        );

        CHECK_THAT
        (
            drop_pressure_matrix(test_result.jacobian),
            Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
        );
    }
}


TEST_CASE("Test gpuOdeSystem (on GPU)", "[GPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-9;


    SECTION("H2"){

        auto test_result = GpuTestKernels::odesystem(TestData::Mechanism::H2);
        auto reference = OFReferenceKernels::odesystem(TestData::Mechanism::H2);


        CHECK_THAT
        (
            drop_pressure(test_result.derivative),
            Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
        );

        CHECK_THAT
        (
            drop_pressure_matrix(test_result.jacobian),
            Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
        );


    }

    SECTION("YAO"){

        auto test_result = GpuTestKernels::odesystem(TestData::Mechanism::YAO);
        auto reference = OFReferenceKernels::odesystem(TestData::Mechanism::YAO);


        CHECK_THAT
        (
            drop_pressure(test_result.derivative),
            Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
        );

        CHECK_THAT
        (
            drop_pressure_matrix(test_result.jacobian),
            Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
        );
    }

    SECTION("GRI"){

        auto test_result = GpuTestKernels::odesystem(TestData::Mechanism::GRI);
        auto reference = OFReferenceKernels::odesystem(TestData::Mechanism::GRI);

        CHECK_THAT
        (
            drop_pressure(test_result.derivative),
            Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
        );

        CHECK_THAT
        (
            drop_pressure_matrix(test_result.jacobian),
            Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
        );


    }
}



static inline bool allFinite(const std::vector<gScalar>& v)
{
    for (const auto e : v)
    {
        if (!std::isfinite(e)) return false;
    }
    return true;
}


TEST_CASE("Test gpuOdeSystem fractional orders at zero concentration (on CPU)", "[CPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-9;

    // CH4^0.2 O2^1.3 with Y_CH4 = 0: d(c^0.2)/dc is singular at c = 0,
    // OpenFOAM-dev evaluates it at max(c, small) (dpowdc)
    const auto mech = TestData::Mechanism::CH4_1STEP;
    const auto y0 = TestData::get_solution_vector(mech);

    auto test_result = CpuTestKernels::odesystem(mech, y0);
    auto reference = OFReferenceKernels::odesystem(mech, y0);

    CHECK(allFinite(test_result.derivative));
    CHECK(allFinite(test_result.jacobian));
    CHECK(allFinite(reference.jacobian));

    CHECK_THAT
    (
        drop_pressure(test_result.derivative),
        Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
    );

    CHECK_THAT
    (
        drop_pressure_matrix(test_result.jacobian),
        Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
    );
}


TEST_CASE("Test gpuOdeSystem temperature limits (on CPU)", "[CPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-9;

    // OpenFOAM-dev clamps a local copy of T to the reaction scheme's
    // [Tlow, Thigh] = [200, 5000] in Standard::derivatives and ::jacobian
    // and uses it for both the thermo and the reaction rates
    const auto mech = TestData::Mechanism::H2;
    const auto nSpecie = TestData::speciesCount(mech);

    for (const gScalar T : {6000.0, 150.0})
    {
        DYNAMIC_SECTION("T = " << T)
        {
            auto y0 = TestData::get_solution_vector(mech);
            y0[nSpecie] = T;

            auto test_result = CpuTestKernels::odesystem(mech, y0);
            auto reference = OFReferenceKernels::odesystem(mech, y0);

            CHECK_THAT
            (
                drop_pressure(test_result.derivative),
                Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
            );

            CHECK_THAT
            (
                drop_pressure_matrix(test_result.jacobian),
                Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
            );
        }
    }
}


TEST_CASE("Test isInteger (on CPU)", "[CPU]")
{
    // Same rule as the OpenFOAM-dev specieExponent(scalar) constructor
    CHECK(!FoamGpu::isInteger(-0.5));
    CHECK(FoamGpu::isInteger(-1.0));
    CHECK(FoamGpu::isInteger(0.0));
    CHECK(FoamGpu::isInteger(2.0));
    CHECK(!FoamGpu::isInteger(0.2));
}


TEST_CASE("Test gpuThirdBodyEfficiencies (on CPU)", "[CPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-12;

    // OpenFOAM-dev 852bc45364 / 385d6e7e64: defaultEfficiency with an
    // efficiencies list or a single efficiency entry
    const Foam::wordList names({"H2", "O2", "H2O", "N2"});
    const Foam::speciesTable species(names);

    const std::vector<gScalar> c = {0.3, 0.7, 1.1, 2.9};
    Foam::scalarField cf(c.size());
    forAll(cf, i)
    {
        cf[i] = c[i];
    }
    Foam::reaction::state s(species.size());
    s.reset(1e5, 1000, cf, 0);

    const std::vector<std::string> dicts =
    {
        "defaultEfficiency 0.5; efficiencies 2((H2O 6.3)(N2 0.5));",
        "defaultEfficiency 2; efficiency (H2O 0);",
        "defaultEfficiency 0.75;",
        "defaultEfficiency 1; efficiencies 4((H2 2)(O2 1)(H2O 6)(N2 1));"
    };

    for (const auto& d : dicts)
    {
        DYNAMIC_SECTION(d)
        {
            const Foam::dictionary dict((Foam::IStringStream(d))());

            const Foam::thirdBodyEfficiencies ref(species, dict);
            const auto test = make_thirdBodyEfficiencies(species, dict);

            const auto cs = make_mdspan(c, extents<1>(c.size()));

            CHECK(test.M(cs) == Approx(ref.M(s)).epsilon(errorTol));

            const auto& dMdc = test.dMdc(cs);
            gScalar M = 0;
            for (size_t i = 0; i < c.size(); ++i)
            {
                CHECK(dMdc[i] == Approx(ref.dMdc()[i]).epsilon(errorTol));
                M += dMdc[i]*c[i];
            }
            CHECK(test.M(cs) == Approx(M).epsilon(errorTol));
        }
    }
}


TEST_CASE("Test gpuOdeSolver (on CPU)", "[CPU]")
{
    using namespace FoamGpu;

    const gScalar xStart = 0.0;
    const gScalar xEnd = 1E-6;
    const gScalar dxTry = 1E-7;




    SECTION("GRI"){

        constexpr double errorTol = 1E-8;


        TestData::Mechanism mech = TestData::Mechanism::GRI;

        SECTION("Rosenbrock12")
        {
            std::string name = "Rosenbrock12";
            auto test_result = drop_pressure(CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);

            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock23")
        {
            std::string name = "Rosenbrock23";
            auto test_result = drop_pressure(CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock34")
        {
            std::string name = "Rosenbrock34";
            auto test_result = drop_pressure(CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }


    }

    SECTION("YAO"){

        constexpr double errorTol = 1E-7;

        TestData::Mechanism mech = TestData::Mechanism::YAO;

        SECTION("Rosenbrock12")
        {
            std::string name = "Rosenbrock12";
            auto test_result = drop_pressure(CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock23")
        {
            std::string name = "Rosenbrock23";
            auto test_result = drop_pressure(CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock34")
        {
            std::string name = "Rosenbrock34";
            auto test_result = drop_pressure(CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }

    }


    SECTION("H2"){

        constexpr double errorTol = 1E-8;


        TestData::Mechanism mech = TestData::Mechanism::H2;

        SECTION("Rosenbrock12")
        {
            std::string name = "Rosenbrock12";
            auto test_result = drop_pressure(CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock23")
        {
            std::string name = "Rosenbrock23";
            auto test_result = drop_pressure(CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock34")
        {
            std::string name = "Rosenbrock34";
            auto test_result = drop_pressure(CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }

    }

}



TEST_CASE("Test memoryResource (on GPU)", "[GPU]")
{

    CHECK(GpuTestKernels::test_memory_resource(TestData::Mechanism::GRI, 101));
    CHECK(GpuTestKernels::test_memory_resource(TestData::Mechanism::GRI, 1032));
    CHECK(GpuTestKernels::test_memory_resource(TestData::Mechanism::GRI, 10031));


}
TEST_CASE("Test gpuOdeSystem fractional orders at zero concentration (on GPU)", "[GPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-9;

    // CH4^0.2 O2^1.3 with Y_CH4 = 0: d(c^0.2)/dc is singular at c = 0,
    // OpenFOAM-dev evaluates it at max(c, small) (dpowdc)
    const auto mech = TestData::Mechanism::CH4_1STEP;
    const auto y0 = TestData::get_solution_vector(mech);

    auto test_result = GpuTestKernels::odesystem(mech, y0);
    auto reference = OFReferenceKernels::odesystem(mech, y0);

    CHECK(allFinite(test_result.derivative));
    CHECK(allFinite(test_result.jacobian));
    CHECK(allFinite(reference.jacobian));

    CHECK_THAT
    (
        drop_pressure(test_result.derivative),
        Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
    );

    CHECK_THAT
    (
        drop_pressure_matrix(test_result.jacobian),
        Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
    );
}

TEST_CASE("Test gpuOdeSystem temperature limits (on GPU)", "[GPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-9;

    // OpenFOAM-dev clamps a local copy of T to the reaction scheme's
    // [Tlow, Thigh] = [200, 5000] in Standard::derivatives and ::jacobian
    // and uses it for both the thermo and the reaction rates
    const auto mech = TestData::Mechanism::H2;
    const auto nSpecie = TestData::speciesCount(mech);

    for (const gScalar T : {6000.0, 150.0})
    {
        DYNAMIC_SECTION("T = " << T)
        {
            auto y0 = TestData::get_solution_vector(mech);
            y0[nSpecie] = T;

            auto test_result = GpuTestKernels::odesystem(mech, y0);
            auto reference = OFReferenceKernels::odesystem(mech, y0);

            CHECK_THAT
            (
                drop_pressure(test_result.derivative),
                Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
            );

            CHECK_THAT
            (
                drop_pressure_matrix(test_result.jacobian),
                Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
            );
        }
    }
}


TEST_CASE("Test gpuOdeSolver (on GPU)", "[GPU]")
{
    using namespace FoamGpu;

    const gScalar xStart = 0.0;
    const gScalar xEnd = 1E-6;
    const gScalar dxTry = 1E-7;



    SECTION("GRI"){

        constexpr double errorTol = 1E-8;

        TestData::Mechanism mech = TestData::Mechanism::GRI;

        SECTION("Rosenbrock12")
        {
            std::string name = "Rosenbrock12";
            auto test_result = drop_pressure(GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock23")
        {
            std::string name = "Rosenbrock23";
            auto test_result = drop_pressure(GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock34")
        {
            std::string name = "Rosenbrock34";
            auto test_result = drop_pressure(GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }

    }
    SECTION("YAO"){

        constexpr double errorTol = 1E-7;

        TestData::Mechanism mech = TestData::Mechanism::YAO;

        SECTION("Rosenbrock12")
        {
            std::string name = "Rosenbrock12";
            auto test_result = drop_pressure(GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock23")
        {
            std::string name = "Rosenbrock23";
            auto test_result = drop_pressure(GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock34")
        {
            std::string name = "Rosenbrock34";
            auto test_result = drop_pressure(GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }

    }



    SECTION("H2"){

        constexpr double errorTol = 1E-8;

        TestData::Mechanism mech = TestData::Mechanism::H2;

        SECTION("Rosenbrock12")
        {
            std::string name = "Rosenbrock12";
            auto test_result = drop_pressure(GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock23")
        {
            std::string name = "Rosenbrock23";
            auto test_result = drop_pressure(GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock34")
        {
            std::string name = "Rosenbrock34";
            auto test_result = drop_pressure(GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry));
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }

    }

}



TEST_CASE("for_each_index (on GPU)", "[GPU]"){

    CHECK(GpuTestKernels::test_for_each_index() == true);

}

TEST_CASE("single_cell_solver (on GPU)", "[GPU]"){

    CHECK(GpuTestKernels::test_single_cell_solver(TestData::GRI, 1) == true);
    CHECK(GpuTestKernels::test_single_cell_solver(TestData::GRI, 2) == true);
    CHECK(GpuTestKernels::test_single_cell_solver(TestData::GRI, 100) == true);


}


/*
TEST_CASE("Test gpuKernelEvaluator (on GPU)", "[GPU]"){

    CHECK(GpuTestKernels::test_evaluator(1) == true);
    CHECK(GpuTestKernels::test_evaluator(2) == true);

}
*/
