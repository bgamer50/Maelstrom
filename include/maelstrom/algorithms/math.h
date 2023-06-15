#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {
    enum binary_op {
        PLUS = 0,
        MINUS = 1,
        TIMES = 2,
        DIVIDE = 3
    };

    /*
        Performs the mathematical operation (+, -, *, or /) elementwise, returning a new vector.
    */
    maelstrom::vector math(maelstrom::vector& left, maelstrom::vector& right, maelstrom::binary_op op);
}