#pragma once

#include "maelstrom/containers/vector.h"

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
        Returns the reduced value and its originating index
        as an any of the input datatype, except
        for the mean reduction, which always returns a double.
    */
    std::pair<boost::any, size_t> reduce(maelstrom::vector& vec, maelstrom::reductor red);
}