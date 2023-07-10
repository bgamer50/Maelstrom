#include "maelstrom/containers/vector.h"
#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/util/cuda_utils.cuh"

namespace maelstrom {
    namespace sparse {
        template<typename T>
        __global__ void k_csr_to_coo(T* csr, T* coo, size_t n_rows) {
            const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;

            for(T k = index; k < n_rows; k += stride) {
                const T start = csr[k];
                const T d = csr[k+1] - csr[k];
                for(T i = 0; i < d; ++i) {
                    coo[start+i] = k;
                }
            }
        }

        template <typename E, typename T>
        maelstrom::vector launch_csr_to_coo_cuda(E exec_policy, maelstrom::vector& ptr, size_t nnz) {
            maelstrom::vector coo(
                ptr.get_mem_type(),
                ptr.get_dtype(),
                nnz
            );
            
            const size_t sz = ptr.size() - 1;
            const size_t num_blocks = maelstrom::cuda::num_blocks(sz, MAELSTROM_DEFAULT_BLOCK_SIZE);

            k_csr_to_coo<<<num_blocks, MAELSTROM_DEFAULT_BLOCK_SIZE>>>(
                static_cast<T*>(ptr.data()),
                static_cast<T*>(coo.data()),
                sz
            );
            cudaDeviceSynchronize();
            maelstrom::cuda::cudaCheckErrors("k_csr_to_coo");

            return coo;
        }

        template <typename E>
        maelstrom::vector csr_to_coo_device_dispatch_val(E exec_policy, maelstrom::vector& ptr, size_t nnz) {
            switch(ptr.get_dtype().prim_type) {
                case UINT64:
                    return launch_csr_to_coo_cuda<E, uint64_t>(exec_policy, ptr, nnz);
                case UINT32:
                    return launch_csr_to_coo_cuda<E, uint32_t>(exec_policy, ptr, nnz);
                case UINT8:
                    return launch_csr_to_coo_cuda<E, uint8_t>(exec_policy, ptr, nnz);
                case INT64:
                    return launch_csr_to_coo_cuda<E, int64_t>(exec_policy, ptr, nnz);
                case INT32:
                    return launch_csr_to_coo_cuda<E, int32_t>(exec_policy, ptr, nnz);
            }

            throw std::runtime_error("Invalid dtype for csr_to_coo");
        }

        template
        maelstrom::vector csr_to_coo_device_dispatch_val(device_exec_t exec_policy, maelstrom::vector& ptr, size_t nnz);
    }
}