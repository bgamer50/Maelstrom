#include <sstream>

#include "test_utils.hpp"
#include "test_utils_dist.cuh"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/dist/shuffle.h"

using namespace maelstrom::test;

void test_dist_select_basic(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    ncclComm_t comm;
    initialize_dist_env(argc, argv, &comm);

    try {
        for(auto storage : {maelstrom::DIST_HOST, maelstrom::DIST_DEVICE, maelstrom::DIST_MANAGED}) {
            test_dist_select_basic(storage);
        }
    } catch(std::exception& err) {
                std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    teardown_dist_env();
    std::cout << "DONE!" << std::endl;
}

void test_dist_select_basic(maelstrom::storage storage) {
    size_t rank = maelstrom::get_rank();
    size_t world_size = maelstrom::get_world_size();

    auto vec = maelstrom::arange(storage, (uint64_t)(8 * world_size));
    
    // select everything on all ranks
    auto s = maelstrom::arange(maelstrom::single_storage_of(storage), (uint64_t)(8 * world_size));
    s = maelstrom::to_dist_vector(std::move(s));

    auto result = maelstrom::select(vec, s);
}