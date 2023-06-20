#pragma once

#include <string>
#include <cuda_runtime.h>

#define MAELSTROM_DEFAULT_BLOCK_SIZE 128ul
#define MAX_NUM_BLOCKS 1024ul

namespace maelstrom {
    namespace cuda {

        inline size_t num_blocks(const size_t num_elements, const size_t num_blocks) {
            return std::min(
                std::max(num_elements / num_blocks, 1ul),
                MAX_NUM_BLOCKS
            );
        }

        inline void cudaCheckErrors(std::string func_name) {
            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess) {
                // print the CUDA error message and exit
                printf("CUDA error calling %s:\n%s\n\n", func_name.c_str(), cudaGetErrorString(error));
                exit(EXIT_FAILURE);
            }
        }
        
    }
}