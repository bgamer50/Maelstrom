#pragma once

#include "containers/vector.h"

namespace maelstrom {
    /*
        Increments elements in the range [start, end) of the given vector by the given value.
    */
    void increment(maelstrom::vector& vec, boost::any inc, size_t start, size_t end);

    /*
        Increments all elements of the given vector by the given value.
    */
    inline void increment(maelstrom::vector& vec, boost::any inc) {
        return increment(vec, inc, 0, vec.size());
    }
}