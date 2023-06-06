#pragma once

#include <tuple>

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/unique.h"
#include "maelstrom/algorithms/reduce.h"

namespace maelstrom {

    /*
        For each key, groups and reduces the values present in the vector, returning
        a reduced result for each key, as well as the global originating index.
        The keys and originating indices end up in their own vectors, obviously.

        max_unique_keys is the largest possible number of unique keys and is required
        to properly allocate memory.

        If sorted=true, the copy-and-sort and re-indexing steps will be skipped.
        If sorted=false, the copy-and-sort and re-indexing steps will be done within this function.
    */
    std::pair<maelstrom::vector, maelstrom::vector> reduce_by_key(maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, size_t max_unique_keys, bool sorted=false);


    /*
        For each key, groups and reduces the values present in the vector, returning
        a reduced result for each key, as well as the global originating index.
        The keys and originating indices end up in their own vectors, obviously.

        Automatically calculates the size of the returned array based on the number of
        unique keys, which may be inefficient.

        If sorted=true (sorted by key), the copy-and-sort and re-indexing steps will be skipped.
        If sorted=false (not sorted by key), the copy-and-sort and re-indexing steps will be done within this function.
    */
    inline std::pair<maelstrom::vector, maelstrom::vector> reduce_by_key(maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, bool sorted=false) {
        return reduce_by_key(
            input_keys,
            input_values,
            red,
            maelstrom::unique(input_keys, sorted).size(),
            sorted
        );
    }
}