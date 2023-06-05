#pragma once

#include "containers/vector.h"

namespace maelstrom {

    enum reductor {
        MIN = 0,
        MAX = 1,
        SUM = 2,
        PRODUCT = 3,
        MEAN = 4
    };

    /*
        Reduces the vector using the given reductor.
        Returns the reduced value as an any.
    */
    boost::any reduce(maelstrom::vector& vec, maelstrom::reductor red);
}