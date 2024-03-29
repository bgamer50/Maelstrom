#include "mpi.h"
#include "nccl.h"
#include <cuda_runtime.h>

#include "maelstrom/storage/dist.cuh"
#include "maelstrom/dist_utils/nccl_utils.cuh"

inline void initialize_dist_env(int argc, char* argv[], ncclComm_t* comm) {
    int mpi_rank, mpi_world_size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

    cudaSetDevice(mpi_rank);

    ncclUniqueId nccl_id;
    if(mpi_rank == 0) {
        ncclGetUniqueId(&nccl_id);
    }
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId) * 1, MPI_BYTE, 0, MPI_COMM_WORLD);

    maelstrom::nccl::ncclCheckErrors(
        ncclCommInitRank(comm, mpi_world_size, nccl_id, mpi_rank),
        "nccl comm init rank"
    );
    cudaDeviceSynchronize();
    std::cout << "rank: " << mpi_rank << std::endl;
    std::cout << "nccl initialized" << std::endl;

    maelstrom::dist_init(mpi_world_size, mpi_rank, comm);
}

inline void teardown_dist_env() {
    auto comm = maelstrom::get_nccl_comms();
    maelstrom::nccl::ncclCheckErrors(
        ncclCommDestroy(comm),
        "nccl comm destroy"
    );

    MPI_Finalize();
}