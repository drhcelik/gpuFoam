#pragma once

// Host emulation of the subset of the HIP runtime used by the GPU test
// kernels: serial execution in host memory. For testing only.

#include <cstddef>
#include <cstdlib>
#include <cstring>

#define __global__

typedef int hipError_t;

enum { hipSuccess = 0 };

enum hipMemcpyKind {
    hipMemcpyDeviceToHost,
    hipMemcpyHostToDevice,
    hipMemcpyDeviceToDevice
};

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1)
        : x(a)
        , y(b)
        , z(c) {}
};

template <class T> inline hipError_t hipMalloc(T** p, size_t n) {
    *p = static_cast<T*>(std::calloc(n ? n : 1, 1));
    return *p ? hipSuccess : 1;
}

inline hipError_t hipFree(void* p) {
    std::free(p);
    return hipSuccess;
}

inline hipError_t hipGetLastError() { return hipSuccess; }

inline hipError_t hipDeviceSynchronize() { return hipSuccess; }

inline hipError_t
hipMemcpy(void* d, const void* s, size_t n, hipMemcpyKind) {
    std::memcpy(d, s, n);
    return hipSuccess;
}

inline const char* hipGetErrorString(hipError_t) { return "host emulation"; }

#define hipLaunchKernelGGL(kernel, grid, block, shmem, stream, ...)          \
    kernel(__VA_ARGS__)
