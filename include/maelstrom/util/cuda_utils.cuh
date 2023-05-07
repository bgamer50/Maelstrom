#pragma once

#include <string>
#include <cuda_runtime.h>

namespace maelstrom {
    namespace cuda {

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