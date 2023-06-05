#pragma once

#include "containers/vector.h"
#include "algorithms/unique.h"
#include <tuple>

namespace maelstrom {

    /*
        Returns a pair.
        The first element of the pair is the list of unique values (dtype same as input vector).
        The second element of the pair is the count of each value (uint64 dtype).

        max_num_values is the maximum number of unique values and is required.
        If sorted=true, will skip the required copy-and-sort step.
    */
    std::pair<maelstrom::vector, maelstrom::vector> count_unique(maelstrom::vector& vec, size_t max_num_values, bool sorted=false);

    /*
        Returns a pair.
        The first element of the pair is the list of unique values (dtype same as input vector).
        The second element of the pair is the count of each value (uint64 dtype).

        Will automatically calculate the maximum number of unique values.
        If sorted=true, will skip the required copy-and-sort step.
    */
    inline std::pair<maelstrom::vector, maelstrom::vector> count_unique(maelstrom::vector& vec, bool sorted=false) {
        size_t unique_count = maelstrom::unique(vec, sorted).size();
        return count_unique(vec, unique_count, sorted);
    }

}