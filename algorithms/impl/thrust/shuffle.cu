#include "maelstrom/algorithms/dist/shuffle.h"
#include "maelstrom/algorithms/count_unique.h"
#include "maelstrom/algorithms/prefix_sum.h"
#include "maelstrom/algorithms/assign.h"

#include "maelstrom/dist_utils/nccl_utils.cuh"

#include <cuda_runtime.h>
#include "nccl.h"

#include <iostream>

namespace maelstrom {
    void shuffle(maelstrom::vector& vec, maelstrom::vector& rix) {
        // FIXME rix should be allowed to be distributed
        auto rix_mem_type = rix.get_mem_type();
        if(maelstrom::is_dist(rix_mem_type)) {
            throw std::runtime_error("Can't set rix to a distributed vector");
        }

        if(rix.local_size() != vec.local_size()) {
            throw std::runtime_error("rix size does not match local vector size");
        }

        size_t rank = maelstrom::get_rank();
        size_t world_size = maelstrom::get_world_size();
        auto comm = maelstrom::get_nccl_comms();
        auto stream = std::any_cast<cudaStream_t>(vec.get_stream());
        auto dtype = vec.get_dtype();

        maelstrom::vector destinations, counts;
        std::tie(destinations, counts) = maelstrom::count_unique(rix, true);
        
        maelstrom::vector send_sizes(maelstrom::MANAGED, maelstrom::uint64, world_size * world_size);
        maelstrom::assign(
            send_sizes,
            destinations,
            counts
        );

        maelstrom::vector send_offsets(send_sizes, false);
        send_offsets.resize(world_size);
        maelstrom::prefix_sum(send_offsets);
        send_offsets = send_offsets.to(maelstrom::HOST);        

        maelstrom::nccl::ncclCheckErrors(
            ncclAllGather(send_sizes.data(), send_sizes.data(), world_size, ncclUint64, comm, stream),
            "shuffle allgather get send sizes"
        );
        cudaStreamSynchronize(stream);

        size_t new_partition_size = 0;
        for(size_t k = 0; k < world_size; ++k) new_partition_size += static_cast<size_t*>(send_sizes.data())[k * world_size + rank];

        // Start group to avoid hang
        size_t* offset_a = static_cast<size_t*>(send_offsets.data());
        size_t bytes_per_element = maelstrom::size_of(dtype);

        maelstrom::nccl::ncclCheckErrors(ncclGroupStart(), "shuffle start nccl group");
        {
            for(size_t dst = 0; dst < world_size; ++dst) {
                size_t start_offset = (dst == 0) ? 0 : offset_a[dst - 1];
                size_t end_offset = offset_a[dst];
                size_t send_size = (end_offset - start_offset) * bytes_per_element;
                
                uint8_t* data_start = static_cast<uint8_t*>(vec.data()) + (start_offset * bytes_per_element);
                maelstrom::nccl::ncclCheckErrors(
                    ncclSend(data_start, send_size, ncclUint8, dst, comm, stream),
                    "shuffle send data"
                );
                cudaStreamSynchronize(stream);
            }

            vec.clear();
            vec.resize_local(new_partition_size);

            size_t received = 0;
            for(size_t src = 0; src < world_size; ++src) {
                size_t recv_bytes = static_cast<size_t*>(send_sizes.data())[src * world_size + rank] * bytes_per_element;

                maelstrom::nccl::ncclCheckErrors(
                    ncclRecv(static_cast<uint8_t*>(vec.data()) + received, recv_bytes, ncclUint8, src, comm, stream),
                    "shuffle recv data"
                );
                cudaStreamSynchronize(stream);

                received += recv_bytes;
            }
        }
        maelstrom::nccl::ncclCheckErrors(ncclGroupEnd(), "shuffle end nccl group");
    }
}