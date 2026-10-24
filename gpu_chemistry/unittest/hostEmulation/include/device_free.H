#pragma once

// Host emulation of gpu_utils/common/device_free.H. For testing only.

#include <cstdlib>

template <class T> static inline void device_free(T* ptr) { std::free(ptr); }
