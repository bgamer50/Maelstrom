#include "maelstrom/storage/storage.h"
#include "maelstrom/storage/dist.cuh"

namespace maelstrom {
    struct maelstrom_dist_env {
        size_t world_size = 0;
        size_t rank = 0;
        ncclComm_t* nccl_comms;
    };

    maelstrom_dist_env dist_env;

    void dist_init(size_t world_size, size_t rank, ncclComm_t* nccl_comms) {
        if(dist_env.world_size > 0) {
            throw std::runtime_error("Error: maelstrom distributed env already initialized");
        }

        dist_env = maelstrom_dist_env{world_size, rank, nccl_comms};
    }
        
    size_t get_world_size() {
        return dist_env.world_size;
    }

    size_t get_rank() {
        return dist_env.rank;
    }

    ncclComm_t& get_nccl_comms() {
        return *dist_env.nccl_comms;
    }

};