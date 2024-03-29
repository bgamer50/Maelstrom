#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {
    /*
        Rebalances the partitions of the given vector to be as evenly
        distributed across GPUs as possible.
    */
    void rebalance(maelstrom::vector& vec);
}