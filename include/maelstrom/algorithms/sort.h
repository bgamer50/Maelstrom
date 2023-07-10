#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {

    maelstrom::vector sort(std::vector<std::reference_wrapper<maelstrom::vector>> vectors);

    /*
        Sorts the vector in place.
        Returns the indices that sorted the given vector.
    */
    inline maelstrom::vector sort(maelstrom::vector& vec) {
        return sort(std::vector({std::ref(vec)}));
    }

}