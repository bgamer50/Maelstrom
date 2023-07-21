#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {

    /*
        Each element of ix corresponds to an index in dst.
        For each element e in ix, dst[e] is assigned to the corresponding value in values.
    */
    void assign(maelstrom::vector& dst, maelstrom::vector& ix, maelstrom::vector& values);

}