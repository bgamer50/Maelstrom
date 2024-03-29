#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {

    /*
        Sets elements in the range [start, end) to the given value.
        Sets in place in the given vector's local partition.
    */
    void set(maelstrom::vector& vec, size_t local_start, size_t local_end, std::any val);

    /*
        Sets all elements of the given vector to the given value.
    */
    inline void set(maelstrom::vector& vec, std::any val) {
        return set(vec, 0, vec.local_size(), val);
    }
}