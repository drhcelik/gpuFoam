R"(reactions
{
    // Reactions of the rate types supported by gpuFoam besides those of the
    // H2, GRI and YAO mechanisms, using the H2 mechanism species. The rate
    // coefficients are not physical.

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
}
Tlow            200;
Thigh           5000;)"
