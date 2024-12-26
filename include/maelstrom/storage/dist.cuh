#pragma once

#include "nccl.h"
#include <cuda_runtime.h>

namespace maelstrom {
    void dist_init(size_t world_size, size_t rank, ncclComm_t* nccl_comms);
    
    size_t get_world_size();
    
    size_t get_rank();

    ncclComm_t& get_nccl_comms();

}