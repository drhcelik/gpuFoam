#pragma once

// Host emulation of gpu_utils/common/error_handling.H: the checked call is
// evaluated. For testing only.

#include "cuda_host_dev.H"
#include "hip_host_emu.H"

#include <cstdio>
#include <cstdlib>
#include <iostream>

#define gpuErrorCheck(call)                                                  \
    do {                                                                     \
        hipError_t gpuErr = call;                                            \
        if (hipSuccess != gpuErr) {                                          \
            printf("GPU Error - %s:%d: '%s'\n",                              \
                   __FILE__,                                                 \
                   __LINE__,                                                 \
                   hipGetErrorString(gpuErr));                               \
            exit(1);                                                         \
        }                                                                    \
    } while (0)
