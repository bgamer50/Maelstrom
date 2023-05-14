#pragma once

#include "containers/vector.h"

namespace maelstrom {

    /*
        Indexes the given vector by the given indices.
        Returns a new vector W, such that for the original
        vector V and index I, W[i] = V[I[i]].
    */
    maelstrom::vector select(maelstrom::vector& vec, maelstrom::vector& idx);
    
}