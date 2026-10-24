R"(reactions
{
    // Reactions of the rate types supported by gpuFoam besides those of the
    // H2, GRI and YAO mechanisms, using the H2 mechanism species. The rate
    // coefficients are not physical.

    // Activation energy Ea [J/kmol] instead of Ta
    reversibleArrheniusEa
    {
        type            reversibleArrhenius;
        reaction        "H2 + O = OH + H";
        A               50.6;
        beta            2.67;
        Ea              2.6316e+07;
    }

    irreversibleThirdBodyArrhenius
    {
        type            irreversibleThirdBodyArrhenius;
        reaction        "2O = O2";
        A               1.2e+11;
        beta            -1;
        Ta              0;
        defaultEfficiency 1;
        efficiencies    3((H2 2.4)(H2O 15.4)(AR 0.83));
    }

    irreversibleArrheniusLindemannFallOff
    {
        type            irreversibleArrheniusLindemannFallOff;
        reaction        "H + O2 = HO2";
        k0
        {
            A               6.366e+14;
            beta            -1.72;
            Ta              264.2;
        }
        kInf
        {
            A               4.65e+09;
            beta            0.44;
            Ta              0;
        }
        F
        {
        }
        thirdBodyEfficiencies
        {
            defaultEfficiency 1;
            efficiencies    3((H2 2)(H2O 14)(O2 0.78));
        }
    }

    irreversibleArrheniusTroeFallOff
    {
        type            irreversibleArrheniusTroeFallOff;
        reaction        "2OH = H2O2";
        k0
        {
            A               2.01e+11;
            beta            -0.584;
            Ta              -1154;
        }
        kInf
        {
            A               1.11e+11;
            beta            -0.37;
            Ta              0;
        }
        F
        {
            alpha           0.7346;
            Tsss            94;
            Ts              1756;
            Tss             5182;
        }
        thirdBodyEfficiencies
        {
            defaultEfficiency 1;
            efficiencies    2((H2 2)(H2O 6));
        }
    }

    reversibleArrheniusSRIFallOff
    {
        type            reversibleArrheniusSRIFallOff;
        reaction        "H + OH = H2O";
        k0
        {
            A               4.0e+16;
            beta            -2;
            Ta              0;
        }
        kInf
        {
            A               1.0e+11;
            beta            0.2;
            Ta              120;
        }
        F
        {
            a               0.45;
            b               797;
            c               979;
            d               1.2;
            e               0.3;
        }
        thirdBodyEfficiencies
        {
            defaultEfficiency 1;
            efficiencies    3((H2 2)(H2O 6.3)(AR 0.38));
        }
    }

    irreversibleArrheniusSRIFallOff
    {
        type            irreversibleArrheniusSRIFallOff;
        reaction        "H + O2 = HO2";
        k0
        {
            A               6.366e+14;
            beta            -1.72;
            Ta              264.2;
        }
        kInf
        {
            A               4.65e+09;
            beta            0.44;
            Ta              0;
        }
        F
        {
            a               0.6;
            b               300;
            c               1500;
            d               0.9;
            e               -0.1;
        }
        thirdBodyEfficiencies
        {
            defaultEfficiency 1;
        }
    }

    reversibleArrheniusLindemannChemicallyActivated
    {
        type            reversibleArrheniusLindemannChemicallyActivated;
        reaction        "H + HO2 = H2O2";
        k0
        {
            A               2.0e+14;
            beta            -1.1;
            Ta              150;
        }
        kInf
        {
            A               3.0e+10;
            beta            0.3;
            Ta              50;
        }
        F
        {
        }
        thirdBodyEfficiencies
        {
            defaultEfficiency 1;
            efficiencies    2((H2 2)(H2O 6));
        }
    }

    irreversibleArrheniusTroeChemicallyActivated
    {
        type            irreversibleArrheniusTroeChemicallyActivated;
        reaction        "OH + OH = H2O + O";
        k0
        {
            A               3.5e+13;
            beta            -0.8;
            Ta              -200;
        }
        kInf
        {
            A               5.0e+09;
            beta            0.1;
            Ta              400;
        }
        F
        {
            alpha           0.6;
            Tsss            100;
            Ts              1500;
            Tss             4000;
        }
        thirdBodyEfficiencies
        {
            defaultEfficiency 1.2;
            efficiency      (N2 0.7);
        }
    }

    reversibleArrheniusSRIChemicallyActivated
    {
        type            reversibleArrheniusSRIChemicallyActivated;
        reaction        "O + H2O = 2OH";
        k0
        {
            A               8.0e+12;
            beta            -0.5;
            Ta              8000;
        }
        kInf
        {
            A               2.0e+08;
            beta            1.1;
            Ta              7500;
        }
        F
        {
            a               0.5;
            b               500;
            c               1200;
            d               1.1;
            e               0.2;
        }
        thirdBodyEfficiencies
        {
            defaultEfficiency 1;
            efficiencies    1((H2O 10));
        }
    }

    reversibleLandauTeller
    {
        type            reversibleLandauTeller;
        reaction        "OH + H2 = H + H2O";
        A               2.14e+05;
        beta            1.52;
        Ta              1735.5;
        B               -2.3;
        C               15.7;
    }

    irreversibleLandauTeller
    {
        type            irreversibleLandauTeller;
        reaction        "O + H2O = 2OH";
        A               4.5e+04;
        beta            2.0;
        Ea              6.1e+07;
        B               3.1;
        C               -4.2;
    }

    reversibleJanev
    {
        type            reversibleJanev;
        reaction        "H + HO2 = 2OH";
        A               7.08e+10;
        beta            0;
        Ta              150;
        b               9(-0.3 0.12 -0.021 0.0035 -4.1e-4 3.2e-5 -1.5e-6 4.1e-8 -4.9e-10);
    }

    irreversibleJanev
    {
        type            irreversibleJanev;
        reaction        "H2O2 + H = H2O + OH";
        A               2.41e+10;
        beta            0.5;
        Ta              1997.7;
        b               9(0.2 -0.05 0.011 -0.0012 7.5e-5 -2.6e-6 4.0e-8 1.0e-10 -6.0e-12);
    }

    // The test pressure 202650 Pa is inside the pressure range
    reversiblePLog
    {
        type            reversiblePLog;
        reaction        "HO2 + H = H2 + O2";
        coeffs
        4
        (
            (1013.25 2.1e+07 1.1 -500)
            (101325 7.5e+08 0.6 120)
            (1013250 3.3e+10 0.1 900)
            (10132500 1.2e+11 -0.2 1500)
        );
    }

    // The test pressure is below the pressure range
    irreversiblePLog
    {
        type            irreversiblePLog;
        reaction        "H2O2 + O = OH + HO2";
        coeffs
        2
        (
            (506625 9.5e+03 2.0 2000)
            (5066250 2.2e+05 1.6 2300)
        );
    }

    // The test pressure is above the pressure range
    reversiblePLogLow
    {
        type            reversiblePLog;
        reaction        "H2O2 + OH = H2O + HO2";
        coeffs
        3
        (
            (1013.25 1.0e+09 0 0)
            (10132.5 1.7e+09 -0.1 30)
            (101325 5.8e+11 -0.9 4800)
        );
    }
}
Tlow            200;
Thigh           5000;)"
