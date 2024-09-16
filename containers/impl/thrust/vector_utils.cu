#include "maelstrom/containers/vector.h"

#include "maelstrom/algorithms/reduce.h"

#include "maelstrom/dist_utils/dist_partition.cuh"
#include <iostream>

namespace maelstrom {

    maelstrom::vector to_dist_vector(maelstrom::vector vec) {
        auto dist_vec = maelstrom::vector(
            maelstrom::dist_storage_of(vec.get_mem_type()),
            vec.get_dtype(),
            vec.data(),
            vec.size(),
            true
        );

        dist_vec.own();
        vec.disown();
        return dist_vec;
    }

}