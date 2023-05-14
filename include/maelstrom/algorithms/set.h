#pragma once

#include "containers/vector.h"

namespace maelstrom {

    /*
        Sets elements in the range [start, end) to the given value.
        Sets in place in the given vector.
    */
    void set(maelstrom::vector& vec, size_t start, size_t end, boost::any val);

    
}