#include "maelstrom/algorithms/dist/rebalance.h"
#include "maelstrom/algorithms/dist/shuffle.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/prefix_sum.h"
#include "maelstrom/algorithms/search_sorted.h"
#include "maelstrom/storage/dist.cuh"

#include "maelstrom/dist_utils/dist_partition.cuh"
#include "maelstrom/dist_utils/nccl_utils.cuh"

#include "nccl.h"
#include <cuda_runtime.h>

namespace maelstrom {

    void rebalance(maelstrom::vector& vec) {
        // quit if this is not a distributed vector
        auto mem_type = vec.get_mem_type();
        if(!maelstrom::is_dist(mem_type)) return;

        size_t rank = maelstrom::get_rank();
        size_t world_size = maelstrom::get_world_size();
        auto comm = maelstrom::get_nccl_comms();
        auto stream = maelstrom::get_cuda_stream();
        auto dtype = vec.get_dtype();

        auto partitions = maelstrom::get_partitions(vec.size());
        maelstrom::vector partitions_vec(
            maelstrom::single_storage_of(mem_type),
            maelstrom::uint64,
            static_cast<size_t*>(partitions.data()) + 1,
            partitions.size() - 1,
            false
        );
        
        size_t local_size = vec.local_size();
        size_t* local_size_device;
        cudaMalloc(&local_size_device, sizeof(size_t) * 1);
        cudaMemcpy(local_size_device, &local_size, sizeof(size_t) * 1, cudaMemcpyDefault);

        maelstrom::vector local_offsets(maelstrom::HOST, maelstrom::uint64, world_size + 1);
        maelstrom::nccl::ncclCheckErrors(
            ncclAllGather(local_size_device, static_cast<size_t*>(local_offsets.data()) + 1, 1, ncclUint64, comm, stream),
            "rebalance allgather get local sizes"
        );
        cudaDeviceSynchronize();
        cudaFree(local_size_device);

        static_cast<size_t*>(local_offsets.data())[0] = 0;
        maelstrom::prefix_sum(local_offsets);

        auto ix = maelstrom::arange(maelstrom::single_storage_of(mem_type), static_cast<size_t*>(local_offsets.data())[rank], static_cast<size_t*>(local_offsets.data())[rank + 1]);
        auto rix = maelstrom::search_sorted(
            partitions_vec,
            ix
        );
        
        // At this point, rix contains the new partitions of each element
        maelstrom::shuffle(vec, rix);
    }

}