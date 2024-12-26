#pragma once

#include "nccl.h"
#include <sstream>

#include "maelstrom/storage/dist.cuh"

namespace maelstrom {
    namespace nccl {

        inline void ncclCheckErrors(ncclResult_t res, std::string func_name) {
            if(res != ncclSuccess) {
                std::stringstream sx;
                sx << "(rank " << maelstrom::get_rank() << ") ";
                sx << "NCCL Error occurred at \"" << func_name << "\": " << ncclGetErrorString(res);
                throw std::runtime_error(sx.str());
            }
        }

    }
}