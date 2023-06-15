#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {

    /*
        Returns the indices of the left vector corresponding to the
        set intersection of the two vectors.
        The output pointed to by the returned index vector will be in sorted order.

        If not sorted, will copy and sort the left and right vectors.
    */
    maelstrom::vector intersection(maelstrom::vector& left, maelstrom::vector& right, bool sorted=false);

}