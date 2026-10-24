#pragma once

// Host emulation of gpu_utils/common/for_each_index.H: op(i) is evaluated
// serially for all i in [0, n[. For testing only.

#include "gpu_constants.H"

template <class UnaryOperation>
static inline void for_each_index(UnaryOperation op, gLabel n) {
    for (gLabel i = 0; i < n; ++i) { op(i); }
}
