#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {
    /*
        Increments elements in the range [start, end) of the given vector by the given value.
        If decrement=true, will decrement by the given value instead of incrementing.
    */
    void increment(maelstrom::vector& vec, boost::any inc, size_t start, size_t end, bool decrement=false);

    /*
        Increments all elements of the given vector by the given value.
        If decrement=true, will decrement by the given value instead of incrementing.
    */
    inline void increment(maelstrom::vector& vec, boost::any inc, bool decrement=false) {
        return increment(vec, inc, 0, vec.size(), decrement);
    }

    /*
        Increments all elements of the given vector by 1.
        If decrement=true, will decrement all elements by 1 instead.
    */
   inline void increment(maelstrom::vector& vec, bool decrement=false) {
        return increment(vec, boost::any(), 0, vec.size(), decrement);
   }
}