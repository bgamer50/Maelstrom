#include "maelstrom/containers/vector.h"
#include "maelstrom/util/any_utils.cuh"
#include "maelstrom/util/cuda_utils.cuh"
#include "maelstrom/thrust_utils/execution.cuh"

namespace maelstrom {
    namespace sparse {

        template <typename T>
        __global__ void k_search_sorted_sparse(T* row, T* col, T* ix_r, T* ix_c, T* output_ix, size_t ix_size, T default_index) {
            const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;

            for(size_t k = index; k < ix_size; k += stride) {
                const T row_idx = ix_r[k];
                const T row_start = row[row_idx];
                const T row_end = row[row_idx+1];

                // Find the first index of the column
                const T value = ix_c[k];

                T left = row_start;
                T right = row_end;
                
                while(left < right) {
                    const T i = (left + right) / 2;
                    const T lower_value = col[i];

                    if(lower_value < value) left = i + 1;
                    else right = i;
                }

                output_ix[k] = (col[left] == value) ? left : default_index;
            }
        }

        template <typename E, typename T>
        maelstrom::vector launch_search_sorted_sparse_device(E exec_policy, maelstrom::vector& row, maelstrom::vector& col, maelstrom::vector& ix_r, maelstrom::vector& ix_c, std::any index_not_found) {
            const size_t sz = ix_r.size();
            const size_t num_blocks = maelstrom::cuda::num_blocks(sz, MAELSTROM_DEFAULT_BLOCK_SIZE);
            
            maelstrom::vector output_ix(row.get_mem_type(), row.get_dtype(), sz);

            k_search_sorted_sparse<<<num_blocks, MAELSTROM_DEFAULT_BLOCK_SIZE>>>(
                static_cast<T*>(row.data()),
                static_cast<T*>(col.data()),
                static_cast<T*>(ix_r.data()),
                static_cast<T*>(ix_c.data()),
                static_cast<T*>(output_ix.data()),
                sz,
                (!index_not_found.has_value()) ? std::numeric_limits<T>::max() : std::any_cast<T>(maelstrom::safe_any_cast(index_not_found, ix_r.get_dtype()))
            );
            cudaDeviceSynchronize();
            maelstrom::cuda::cudaCheckErrors("k_search_sorted_sparse");

            return output_ix;
        }

        template <typename E>
        maelstrom::vector search_sorted_sparse_device_dispatch_ix(E exec_policy, maelstrom::vector& row, maelstrom::vector& col, maelstrom::vector& ix_r, maelstrom::vector& ix_c, std::any index_not_found) {
            switch(row.get_dtype().prim_type) {
                case UINT64:
                    return launch_search_sorted_sparse_device<E, uint64_t>(exec_policy, row, col, ix_r, ix_c, index_not_found);
                case UINT32:
                    return launch_search_sorted_sparse_device<E, uint32_t>(exec_policy, row, col, ix_r, ix_c, index_not_found);
                case UINT8:
                    return launch_search_sorted_sparse_device<E, uint8_t>(exec_policy, row, col, ix_r, ix_c, index_not_found);
                case INT64:
                    return launch_search_sorted_sparse_device<E, int64_t>(exec_policy, row, col, ix_r, ix_c, index_not_found);
                case INT32:
                    return launch_search_sorted_sparse_device<E, int32_t>(exec_policy, row, col, ix_r, ix_c, index_not_found);
            }

            throw std::runtime_error("Unsupporte data type for index in search_sorted_sparse_device");
        }

        template
        maelstrom::vector search_sorted_sparse_device_dispatch_ix(maelstrom::device_exec_t exec_policy, maelstrom::vector& row, maelstrom::vector& col, maelstrom::vector& ix_r, maelstrom::vector& ix_c, std::any index_not_found);

    }
}
