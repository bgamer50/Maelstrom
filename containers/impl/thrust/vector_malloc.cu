#include "maelstrom/containers/vector.h"
#include "maelstrom/util/cuda_utils.cuh"

#include <cuda_runtime.h>

#include <sstream>
#include <iostream>

namespace maelstrom {

    void* maelstrom::vector::alloc(size_t N) {
        size_t dtype_size = maelstrom::size_of(this->dtype);

        switch(this->mem_type) {
            case HOST: {
                void* ptr;
                cudaMallocManaged(&ptr, dtype_size * N);
                cudaMemAdvise(ptr, dtype_size * N, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
                cudaDeviceSynchronize();
                maelstrom::cuda::cudaCheckErrors("vector alloc device memory");
                return ptr;
            }
            case DEVICE: {
                void* ptr;
                cudaMalloc(&ptr, dtype_size * N);
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
        switch(this->mem_type) {
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
                cudaFree(ptr);
                cudaDeviceSynchronize();
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

        cudaMemcpy(dst, src, maelstrom::size_of(this->dtype) * size, cudaMemcpyDefault);
        cudaDeviceSynchronize();
        maelstrom::cuda::cudaCheckErrors("TypeErasedVector copy");
    }

}