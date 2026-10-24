#pragma once

// Host emulation of gpu_utils/common/host_device_transfers.H: the "device"
// memory is host memory. For testing only.

#include <algorithm>
#include <iterator>
#include <type_traits>

template <class InputIter, class OutputIter>
static inline void
host_to_device(InputIter h_begin, InputIter h_end, OutputIter d_begin) {
    std::copy(h_begin, h_end, d_begin);
}

template <class InputIter, class OutputIter>
static inline void
device_to_host(InputIter d_begin, InputIter d_end, OutputIter h_begin) {
    std::copy(d_begin, d_end, h_begin);
}
