#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {
    
    /*
        For each element k in the given vector, creates a new vector of size 1
        whose sole element is element k.
    */
    std::vector<maelstrom::vector> unpack(maelstrom::vector& vec);

}