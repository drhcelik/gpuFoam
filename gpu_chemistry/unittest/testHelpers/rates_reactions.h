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
}
Tlow            200;
Thigh           5000;)"
