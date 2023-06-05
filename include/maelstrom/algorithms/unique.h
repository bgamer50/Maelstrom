#pragma once

#include "containers/vector.h"

namespace maelstrom {

    /*
        Returns a maelstrom vector of the indices of the unique values
        in the given array (UINT64 type).  If sorted=true, the copy-and-sort step
        is skipped prior to determining the unique indices.  Otherwise,
        the given vector is copied and sorted first.

        The returned indices correspond to those in the original vector.
    */
    maelstrom::vector unique(maelstrom::vector& vec, bool sorted=false);

}