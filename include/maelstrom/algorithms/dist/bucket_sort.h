#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {
    
    enum sort_partition_type_t {
        ORIGINAL = 0, // preserves original partition sizes
        BALANCED = 1, // rebalances the output partition sizes
        UNIQUE = 2, // each value will only be present on one partition
    };

    /*
        Performs a bucket sort in place on a distributed maelstrom vector.
        Respects the given partition behavior, defaulting to UNIQUE, which will
        not perform any repartitioning after the sort and guarantees that the values
        on each partition do not overlap.
        Returns the indices used for sorting.
    */
    maelstrom::vector bucket_sort(maelstrom::vector& vec, sort_partition_type_t partition_behavior=maelstrom::UNIQUE);

}