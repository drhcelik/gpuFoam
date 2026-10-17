R"(reactions
{
    // Global one-step methane oxidation with fractional (FORD) orders
    // CH4^0.2 O2^1.3 (Westbrook & Dryer 1981 style)
    globalCH4
    {
        type            irreversibleArrhenius;
        reaction        "CH4^0.2 + 2O2^1.3 = CO2 + 2H2O";
        A               2.119e+11;
        beta            0;
        Ta              24379;
    }
}
Tlow            200;
Thigh           5000;)"
