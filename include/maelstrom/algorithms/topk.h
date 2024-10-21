#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {

    /*
        Returns the indices of the top k elements in the vector.
        If descending=true, then this returns the bottom k elements.
    */
    maelstrom::vector topk(maelstrom::vector& vec, size_t k, bool descending=false);

}