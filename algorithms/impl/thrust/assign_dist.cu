#include "maelstrom/containers/vector.h"

#include "maelstrom/algorithms/prefix_sum.h"
#include "maelstrom/algorithms/search_sorted.h"
#include "maelstrom/algorithms/increment.h"
#include "maelstrom/algorithms/assign.h"

#include "maelstrom/algorithms/dist/shuffle.h"

#include "maelstrom/dist_utils/dist_partition.cuh"

#include "nccl.h"
#include <iostream>

namespace maelstrom {
    void assign_dispatch_dist(maelstrom::vector& dst, maelstrom::vector& ix, maelstrom::vector& values) {
        // ix and values must have the same partition scheme
        if(!dst.is_dist() || !ix.is_dist() || !values.is_dist()) {
            throw std::runtime_error("all vectors must be distributed for distributed assign");
        }
        if(ix.local_size() != values.local_size()) {
            throw std::runtime_error("ix and values partitions must match");
        }

        auto mem_type = maelstrom::single_storage_of(dst.get_mem_type());

        auto local_sizes = maelstrom::get_current_partition_sizes(dst);
        maelstrom::prefix_sum(local_sizes);
        auto rix = maelstrom::search_sorted(
            local_sizes,
            ix
        );
        auto rix_local_view = maelstrom::local_view_of(rix);

        auto values_copy = maelstrom::vector(values, false);
        auto ix_copy = maelstrom::vector(ix, false);

        maelstrom::shuffle(ix_copy, rix_local_view);
        maelstrom::shuffle(values_copy, rix_local_view);

        size_t rank = maelstrom::get_rank();
        size_t local_offset = (rank == 0) ? 0 : std::any_cast<size_t>(local_sizes.get(rank - 1));
        
        maelstrom::increment(
            ix_copy,
            local_offset,
            maelstrom::DECREMENT
        );

        auto dst_local_view = maelstrom::local_view_of(dst);
        maelstrom::assign(
            dst_local_view,
            ix_copy,
            values_copy
        );
    }
}