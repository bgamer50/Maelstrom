#include "maelstrom/containers/vector.h"
#include "maelstrom/util/cuda_utils.cuh"

#include <cuda_runtime.h>

#include <sstream>
#include <iostream>

namespace maelstrom {

    void* maelstrom::vector::alloc(size_t N) {
        size_t dtype_size = maelstrom::size_of(this->dtype);

        // Calls the base allocator
        auto base_mem_type = maelstrom::single_storage_of(this->mem_type);

        switch(base_mem_type) {
            case HOST: {
                void* ptr;
                cudaMallocManaged(&ptr, dtype_size * N);
                cudaMemAdvise(ptr, dtype_size * N, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
                cudaDeviceSynchronize();
                maelstrom::cuda::cudaCheckErrors("vector alloc host memory");
                return ptr;
            }
            case DEVICE: {
                void* ptr;
                cudaMallocAsync(&ptr, dtype_size * N, std::any_cast<cudaStream_t>(this->stream));
                cudaDeviceSynchronize();
                maelstrom::cuda::cudaCheckErrors("vector alloc device memory");
                return ptr;
            }
            case MANAGED: {
                void* ptr;
                cudaMallocManaged(&ptr, dtype_size * N);
                cudaDeviceSynchronize();
                std::stringstream sx;
                sx << "vector alloc managed memory (" << this->name << ")";
                maelstrom::cuda::cudaCheckErrors(sx.str());
                return ptr;
            }
            case PINNED: {
                void* ptr;
                cudaMallocHost(&ptr, dtype_size * N);
                cudaDeviceSynchronize();
                maelstrom::cuda::cudaCheckErrors("vector alloc pinned memory");
                return ptr;
            }
        }

        throw std::runtime_error("Invalid memory type provided to vector alloc()");
    }

    void maelstrom::vector::dealloc(void* ptr) {
        if(ptr == nullptr) throw std::invalid_argument("Cannot deallocate a null pointer");

        // Calls the base allocator
        auto base_mem_type = maelstrom::single_storage_of(this->mem_type);
        auto current_stream = std::any_cast<cudaStream_t>(this->stream);

        switch(base_mem_type) {
            case HOST: {
                cudaFree(ptr);
                cudaDeviceSynchronize();
                std::stringstream sx;
                sx << "vector dealloc host-advised managed memory (" << this->name << ")";
                maelstrom::cuda::cudaCheckErrors(sx.str());
                return;
            }
            case MANAGED: {
                cudaFree(ptr);
                cudaDeviceSynchronize();
                std::stringstream sx;
                sx << "vector dealloc managed memory (" << this->name << ")";
                maelstrom::cuda::cudaCheckErrors(sx.str());
                return;
            }
            case DEVICE: {
                cudaFreeAsync(ptr, current_stream);
                cudaStreamSynchronize(current_stream);
                std::stringstream sx;
                sx << "vector dealloc device memory (" << this->name << ")";
                maelstrom::cuda::cudaCheckErrors(sx.str());
                return;
            }
            case PINNED: {
                cudaFreeHost(ptr);
                cudaDeviceSynchronize();
                std::stringstream sx;
                sx << "vector dealloc pinned memory (" << this->name << ")";
                maelstrom::cuda::cudaCheckErrors(sx.str());
                return;
            }
        }

        throw std::runtime_error("Invalid memory type provided to vector dealloc");
    }

    // Copies from src (first arg) to dst (second arg) using cudaMemcpy.
    void maelstrom::vector::copy(void* src, void* dst, size_t size) {
        if(src == dst) return;
        auto current_stream = std::any_cast<cudaStream_t>(this->stream);

        cudaMemcpyAsync(dst, src, maelstrom::size_of(this->dtype) * size, cudaMemcpyDefault, current_stream);
        cudaStreamSynchronize(current_stream);
        maelstrom::cuda::cudaCheckErrors("maelstrom vector copy");
    }

    void maelstrom::vector::pin() {
        if(!this->is_view() || this->mem_type != maelstrom::HOST) throw std::domain_error("Vector must be a host view to be pinned!");

        cudaHostRegister(this->data_ptr, maelstrom::size_of(this->dtype) * this->local_size(), cudaHostRegisterReadOnly);
        cudaDeviceSynchronize();
        maelstrom::cuda::cudaCheckErrors("maelstrom vector pin");
    }

    void maelstrom::vector::unpin() {
        if(!this->is_view() || this->mem_type != maelstrom::HOST) throw std::domain_error("Vector must be a host view to be unpinned!");

        cudaHostUnregister(this->data_ptr);
        cudaDeviceSynchronize();
        maelstrom::cuda::cudaCheckErrors("maelstrom vector unpin");
    }

}