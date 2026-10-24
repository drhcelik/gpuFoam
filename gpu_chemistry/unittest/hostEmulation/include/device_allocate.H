#pragma once

// Host emulation of gpu_utils/common/device_allocate.H: the "device" memory
// is zero-initialised host memory. For testing only.

#include <cstddef>
#include <cstdlib>

template <class T> static inline T* device_allocate(size_t length) {
    return static_cast<T*>(std::calloc(length ? length : 1, sizeof(T)));
}
