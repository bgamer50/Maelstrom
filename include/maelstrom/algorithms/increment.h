#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {
    enum inc_op {
        INCREMENT=0,
        DECREMENT=1
    };

    /*
        Increments elements in the range [start, end) of the given vector by the given value.
        If decrement=true, will decrement by the given value instead of incrementing.
    */
    void increment(maelstrom::vector& vec, std::any inc, size_t start, size_t end, maelstrom::inc_op op=INCREMENT);

    /*
        Increments all elements of the given vector by the given value.
        If decrement=true, will decrement by the given value instead of incrementing.
    */
    inline void increment(maelstrom::vector& vec, std::any inc, maelstrom::inc_op op=INCREMENT) {
        return increment(vec, inc, 0, vec.size(), op);
    }

    /*
        Increments all elements of the given vector by 1.
        If decrement=true, will decrement all elements by 1 instead.
    */
   inline void increment(maelstrom::vector& vec, maelstrom::inc_op op=INCREMENT) {
        return increment(vec, std::any(), 0, vec.size(), op);
   }
}