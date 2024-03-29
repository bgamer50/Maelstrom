#pragma once

#include <numeric>

#include "maelstrom/storage/storage.h"
#include "maelstrom/storage/dist.cuh"
#include "maelstrom/containers/vector.h"

#include "maelstrom/algorithms/set.h"

#include "maelstrom/dist_utils/nccl_utils.cuh"

#include "nccl.h"

namespace maelstrom {
    inline std::vector<size_t> get_partitions(size_t array_length) {
        const size_t world_size = maelstrom::get_world_size();
        size_t q = array_length / world_size;
        size_t r = array_length % world_size;

        std::vector<size_t> dv;
        dv.push_back(0);
        for(size_t k = 0; k < r; ++k) dv.push_back(q+1);
        for(size_t k = 0; k < (q-r); ++k) dv.push_back(q);

        std::partial_sum(dv.begin(), dv.end(), dv.begin());
        return dv;
    }

    inline std::pair<size_t, size_t> get_local_partition(size_t array_length) {
        size_t rank = maelstrom::get_rank();
        auto p = get_partitions(array_length);
        return std::make_pair(
            p[rank],
            p[rank+1]
        );
    }

    inline size_t get_local_partition_size(size_t array_length) {
        size_t a, b;
        std::tie(a, b) = get_local_partition(array_length);

        return b - a;
    }

    inline maelstrom::vector get_current_partition_sizes(maelstrom::vector& vec) {
        maelstrom::vector local_sizes(
            maelstrom::single_storage_of(vec.get_mem_type()),
            maelstrom::uint64,
            maelstrom::get_world_size()
        );
        maelstrom::set(local_sizes, 0, 1, vec.local_size());

        maelstrom::nccl::ncclCheckErrors(
            ncclAllGather(
                local_sizes.data(),
                local_sizes.data(),
                1,
                ncclUint64,
                maelstrom::get_nccl_comms(),
                maelstrom::get_cuda_stream()
            ),
            "get local sizes allgather"
        );
        cudaStreamSynchronize(maelstrom::get_cuda_stream());

        return local_sizes;
    }
}