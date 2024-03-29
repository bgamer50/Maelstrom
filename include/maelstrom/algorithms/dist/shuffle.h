#pragma once

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/set.h"

namespace maelstrom {
    /*
        For the given vector vec and partition list rix,
        shuffles the vector so that each element vec[i] is moved
        to the new partition specified by rix[i].
        Preserves order.
    */
    void shuffle(maelstrom::vector& vec, maelstrom::vector& rix);

    /*
        Shuffles the vector so that all entries are on the given rank.
    */
    inline void shuffle_to_rank(maelstrom::vector& vec, size_t rank) {
        auto mem_type = vec.get_mem_type();
        maelstrom::vector zeros(
            maelstrom::single_storage_of(mem_type),
            maelstrom::uint64,
            vec.local_size()
        );
        maelstrom::set(zeros, static_cast<size_t>(0));

        maelstrom::shuffle(vec, zeros);
    }
}