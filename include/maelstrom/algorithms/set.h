#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {

    /*
        Sets elements in the range [start, end) to the given value.
        Sets in place in the given vector.
    */
    void set(maelstrom::vector& vec, size_t start, size_t end, std::any val);

    /*
        Sets all elements of the given vector to the given value.
    */
    inline void set(maelstrom::vector& vec, std::any val) {
        return set(vec, 0, vec.size(), val);
    }
}