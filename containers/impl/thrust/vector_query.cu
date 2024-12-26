#include "maelstrom/storage/storage.h"
#include "maelstrom/storage/dist.cuh"
#include "maelstrom/containers/vector.h"

#include "maelstrom/dist_utils/dist_partition.cuh"
#include "maelstrom/dist_utils/nccl_utils.cuh"
#include "maelstrom/util/cuda_utils.cuh"

#include "nccl.h"
#include <cuda_runtime.h>
#include <iostream>

namespace maelstrom {

    size_t vector::size() {
        // simply return the filled size for a non-distributed vector
        if(!maelstrom::is_dist(this->mem_type)) return this->filled_size;

        auto current_stream = std::any_cast<cudaStream_t>(this->get_stream());

        // For a distributed vector, take the sum of all filled sizes
        size_t* size_device;
        cudaMalloc(&size_device, sizeof(size_t) * 1);
        cudaMemcpy(size_device, &this->filled_size, sizeof(size_t) * 1, cudaMemcpyDefault);
        cudaStreamSynchronize(current_stream);
        maelstrom::cuda::cudaCheckErrors("initialize size");

        maelstrom::nccl::ncclCheckErrors(
            ncclAllReduce(
            size_device,
            size_device,
            1,
            ncclUint64,
            ncclSum,
            maelstrom::get_nccl_comms(),
            current_stream
            ),
            "sum size all reduce"
        );

        cudaStreamSynchronize(current_stream);

        size_t total_size;
        cudaMemcpy(&total_size, size_device, sizeof(size_t) * 1, cudaMemcpyDefault);
        cudaStreamSynchronize(current_stream);
        cudaFree(size_device);
        return total_size;
    }

    void vector::reserve_local(size_t N) {
        if(N < this->filled_size) throw std::runtime_error("Cannot reserve fewer elements than size!");
        
        if(N <= this->reserved_size) return;

        size_t old_filled_size = this->filled_size;
        this->resize_local(N);
        this->filled_size = old_filled_size;
    }

    void vector::reserve(size_t N) {
        // TODO does this function even make sense for a distributed vector?
        if(maelstrom::is_dist(this->mem_type)) N = maelstrom::get_local_partition_size(N);

        this->reserve_local(N);
    }

    void vector::resize_local(size_t N) {
        if(N == 0) {
            return this->clear();
        }

        if(this->view) throw std::runtime_error("Cannot resize a view!");

        bool empty = (this->reserved_size == 0);
        
        // Don't resize if there is already enough space reserved
        if(N <= reserved_size) {
            this->filled_size = N;
            return;
        }

        void* new_data = this->alloc(N);
        if(!empty) {
            this->copy(this->data_ptr, new_data, this->filled_size);
            this->dealloc(this->data_ptr);
        }
        
        this->data_ptr = new_data;
        this->filled_size = N;
        this->reserved_size = N;
    }

    void vector::resize(size_t N) {
        // FIXME requires rebalancing
        if(maelstrom::is_dist(this->mem_type)) {
            N = maelstrom::get_local_partition_size(N);
            if(this->size() > 0) {
                throw std::runtime_error("Resizing a non-empty vector is currently not supported for distributed vectors");
            }
        }

        this->resize_local(N);
    }

}