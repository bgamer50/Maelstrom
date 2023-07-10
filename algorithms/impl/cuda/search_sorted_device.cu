#include "maelstrom/algorithms/search_sorted.h"
#include "maelstrom/thrust_utils/thrust_utils.cuh"
#include "maelstrom/thrust_utils/execution.cuh"

#include "maelstrom/util/cuda_utils.cuh"

namespace maelstrom {

    template<typename T>
    __global__ void k_search_sorted(T* sorted_array, T* values_to_find, size_t* output, size_t N_sorted_values, size_t N_values_to_find) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = blockDim.x * gridDim.x;

        for(size_t k = idx; k < N_values_to_find; k += stride) {
            const T value = values_to_find[k];

            size_t left = 0;
            size_t right = N_sorted_values;
            
            while(left < right) {
                const size_t i = (left + right) / 2;
                const T lower_value = sorted_array[i];

                if(lower_value <= value) left = i + 1;
                else right = i;

            }

            output[k] = left;
        }
    }

    template <typename E, typename T>
    maelstrom::vector launch_search_sorted_cuda(E exec_policy, maelstrom::vector& sorted_array, maelstrom::vector& values_to_find) {
        maelstrom::vector output(sorted_array.get_mem_type(), uint64, values_to_find.size());

        const size_t sz = values_to_find.size();
        const size_t num_blocks = maelstrom::cuda::num_blocks(sz, MAELSTROM_DEFAULT_BLOCK_SIZE);

        k_search_sorted<<<num_blocks, MAELSTROM_DEFAULT_BLOCK_SIZE>>>(
            static_cast<T*>(sorted_array.data()),
            static_cast<T*>(values_to_find.data()),
            static_cast<size_t*>(output.data()),
            sorted_array.size(),
            sz
        );
        cudaDeviceSynchronize();
        maelstrom::cuda::cudaCheckErrors("k_search_sorted");

        return output;
    }

    template <typename E>
    maelstrom::vector search_sorted_device_dispatch_val(E exec_policy, maelstrom::vector& sorted_array, maelstrom::vector& values_to_find) {
        switch(sorted_array.get_dtype().prim_type) {
            case UINT64:
                return launch_search_sorted_cuda<E, uint64_t>(exec_policy, sorted_array, values_to_find);
            case UINT32:
                return launch_search_sorted_cuda<E, uint32_t>(exec_policy, sorted_array, values_to_find);
            case UINT8:
                return launch_search_sorted_cuda<E, uint8_t>(exec_policy, sorted_array, values_to_find);
            case INT64:
                return launch_search_sorted_cuda<E, int64_t>(exec_policy, sorted_array, values_to_find);
            case INT32:
                return launch_search_sorted_cuda<E, int32_t>(exec_policy, sorted_array, values_to_find);
            case INT8:
                return launch_search_sorted_cuda<E, int8_t>(exec_policy, sorted_array, values_to_find);
            case FLOAT64:
                return launch_search_sorted_cuda<E, double>(exec_policy, sorted_array, values_to_find);
            case FLOAT32:
                return launch_search_sorted_cuda<E, float>(exec_policy, sorted_array, values_to_find);
        }

        throw std::runtime_error("Invalid data type passed to search_sorted");
    }

    template
    maelstrom::vector search_sorted_device_dispatch_val(maelstrom::device_exec_t exec_policy, maelstrom::vector& sorted_array, maelstrom::vector& values_to_find);

}