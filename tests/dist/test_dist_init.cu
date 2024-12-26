#include <sstream>

#include "mpi.h"
#include "nccl.h"

#include <cuda_runtime.h>

#include "test_utils.hpp"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/dist/rebalance.h"
#include "maelstrom/storage/dist.cuh"
#include "maelstrom/dist_utils/nccl_utils.cuh"

using namespace maelstrom::test;

void test_dist_init_basic(int rank, int mpi_world_size);

int main(int argc, char* argv[]) {
    int mpi_rank, mpi_world_size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

    // FIXME make this work with more than 2 workers
    if(mpi_world_size > 2) {
        throw std::runtime_error("test_dist_init only runs with 2 workers!");
    }

    cudaSetDevice(mpi_rank);

    try {
        test_dist_init_basic(mpi_rank, mpi_world_size);
    } catch(std::exception& err) {
                std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    MPI_Finalize();
    std::cout << "DONE!" << std::endl;
}

void test_dist_init_basic(const int rank, const int mpi_world_size) {
    ncclUniqueId nccl_id;
    if(rank == 0) {
        maelstrom::nccl::ncclCheckErrors(
            ncclGetUniqueId(&nccl_id),
            "nccl get unique id"
        );
    }
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId) * 1, MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    maelstrom::nccl::ncclCheckErrors(
        ncclCommInitRank(&comm, mpi_world_size, nccl_id, rank),
        "nccl comm init rank"
    );

    maelstrom::dist_init(mpi_world_size, rank, &comm);

    assert (
        maelstrom::get_rank() == rank
    );

    assert(
        maelstrom::get_world_size() == mpi_world_size
    );

    auto v = maelstrom::arange(maelstrom::DIST_MANAGED, (int)3, (int)11);
    std::cout << "created arange vector" << std::endl;
    std::cout << "size: " << v.size() << std::endl;

    for(size_t k = 0; k < 4; ++k) std::cout << k << ": " << std::any_cast<int>(v.get_local(k)) << std::endl;

    if(rank == 0) {
        std::vector<int> x = {3, 4, 5, 6};
        assert( v.size() == 8 );
        assert_array_equals(static_cast<int*>(v.data()), x.data(), v.local_size());
    }

    if(rank == 1) {
        std::vector<int> x = {7, 8, 9, 10};
        assert( v.size() == 8 );
        assert_array_equals(static_cast<int*>(v.data()), x.data(), v.local_size());
    }

    auto w = maelstrom::vector(maelstrom::DIST_MANAGED, maelstrom::int32);
    if(rank == 0) {
        auto n = maelstrom::arange(maelstrom::HOST, 0, 2);
        w.insert_local(n);
    }
    if(rank == 1) {
        auto n = maelstrom::arange(maelstrom::HOST, 2, 6);
        w.insert_local(n);
    }

    cudaDeviceSynchronize();
    maelstrom::rebalance(w);

    assert( w.size() == 6 );
    assert( w.local_size() == 3 );

    for(size_t k = 0; k < 3; ++k) std::cout << std::any_cast<int>(w.get_local(k)) << " ";
    std::cout << std::endl;
    if(rank == 0) {
        std::vector<int> z = {0, 1, 2};
        assert_array_equals(static_cast<int*>(w.data()), z.data(), 3);
    }
    if(rank == 1) {
        std::vector<int> z = {3, 4, 5};
        assert_array_equals(static_cast<int*>(w.data()), z.data(), 3);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    maelstrom::nccl::ncclCheckErrors(
        ncclCommDestroy(comm),
        "nccl comm destroy"
    );
}