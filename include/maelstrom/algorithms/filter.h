#pragma once

#include "maelstrom/storage/comparison.h"
#include "maelstrom/containers/vector.h"

namespace maelstrom {
    /*
        Returns a new vector containing the indices for which the comparator returned true.
        Does not modify the original vector.
        The returned vector always has dtype UINT64.
    */
    maelstrom::vector filter(maelstrom::vector& vec, maelstrom::comparator cmp, boost::any cmp_val);
}
