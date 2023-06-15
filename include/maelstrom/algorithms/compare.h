#pragma once

#include "maelstrom/storage/comparison.h"
#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/remove_if.h"

namespace maelstrom {

    /*
        Compares each element of the two vectors.  Returns a vector of uint8_t type where 0 corresponds
        to a false predicate, and 1 corresponds to a true predicate.

        If invert=true, the element values are flipped (1 = false and 0 = true)
    */
    maelstrom::vector compare(maelstrom::vector& vec1, maelstrom::vector& vec2, maelstrom::comparator cmp, bool invert=false);

    inline maelstrom::vector compare_select(maelstrom::vector& vec1, maelstrom::vector& vec2, maelstrom::comparator cmp) {
        maelstrom::vector mask = maelstrom::compare(vec1, vec2, cmp, true);

        maelstrom::vector vec1_copy(vec1);
        maelstrom::remove_if(vec1_copy, mask);

        return vec1_copy;
    }

}