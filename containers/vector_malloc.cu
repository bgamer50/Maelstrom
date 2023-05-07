#include "containers/vector.h"
#include "util/cuda_utils.cuh"

#include <cuda_runtime.h>

#include <sstream>

namespace maelstrom {

    void* maelstrom::vector::alloc(size_t N) {
        size_t dtype_size = maelstrom::size_of(this->dtype);

        switch(this->mem_type) {
            case HOST: {
                return static_cast<void*>(new char[N * dtype_size]);
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
                delete static_cast<char*>(ptr);
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
                maelstrom::cuda::cudaCheckErrors("vector dealloc device memory");
                return;
            }
            case PINNED: {
                cudaFreeHost(ptr);
                cudaDeviceSynchronize();
                maelstrom::cuda::cudaCheckErrors("vector dealloc pinned memory");
                return;
            }
        }

        throw std::runtime_error("Invalid memory type provided to vector dealloc");
    }

    // Copies from src (first arg) to dst (second arg) using cudaMemcpy.
    void maelstrom::vector::copy(void* src, void* dst, size_t size) {
        cudaMemcpy(dst, src, maelstrom::size_of(this->dtype) * size, cudaMemcpyDefault);
        maelstrom::cuda::cudaCheckErrors("TypeErasedVector copy");
    }

}