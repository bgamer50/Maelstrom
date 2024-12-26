#pragma once

#include "maelstrom/containers/vector.h"
#include <optional>

namespace maelstrom {

    /*
        Returns a vector of indices that can be used to randomly sample an array of array_size.
        Each element in the subsample is unique (the same element can't be picked more than once).
        If array_size == num_to_select, then the returned array is a permutation vector that
        randomly reorders an array of array_size.
    */
    maelstrom::vector randperm(maelstrom::storage mem_type, size_t array_size, size_t num_to_select, std::optional<size_t> seed);

    /*
        Returns a vector of indices that can be used to randomly sample an array of array_size.
        Each element in the subsample is unique (the same element can't be picked more than once).
        If array_size == num_to_select, then the returned array is a permutation vector that
        randomly reorders an array of array_size.
    */
    inline maelstrom::vector randperm(maelstrom::storage mem_type, size_t array_size, size_t num_to_select) {
        return randperm(mem_type, array_size, num_to_select, std::nullopt);
    }

}