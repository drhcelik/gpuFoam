#pragma once

#include "gpu_constants.H"
#include <string>

namespace FoamGpu {


struct gpuODESolverInputs
{
    //name of the solver
    std::string             name;
    //base class inputs
    gScalar                 absTol;
    gScalar                 relTol;
    gLabel                  maxSteps;
    //adaptive solver inputs
    gScalar safeScale;
    gScalar alphaInc;
    gScalar alphaDec;
    gScalar minScale;
    gScalar maxScale;
    //semi-implicit (Rosenbrock) solver inputs
    //- Use the scaled partial pivoting LU decomposition, otherwise the
    //  unpivoted decomposition with row filtering (OpenFOAM-dev default)
    bool pivot = false;
    //- Row multiplier tolerance of the unpivoted decomposition
    gScalar rowTol = 0;
};

}