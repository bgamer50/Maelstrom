#include <sstream>

#include "test_utils.hpp"
#include "test_utils_dist.cuh"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/dist/shuffle.h"

using namespace maelstrom::test;

void test_dist_set_basic(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    ncclComm_t comm;
    initialize_dist_env(argc, argv, &comm);

    try {
        for(auto storage : {maelstrom::DIST_HOST, maelstrom::DIST_DEVICE, maelstrom::DIST_MANAGED}) {
            test_dist_set_basic(storage);
        }
    } catch(std::exception& err) {
                std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    teardown_dist_env();
    std::cout << "DONE!" << std::endl;
}

void test_dist_set_basic(maelstrom::storage storage) {
    size_t rank = maelstrom::get_rank();

    maelstrom::vector vec(storage, maelstrom::uint32);
    vec.resize_local(16);
    assert( vec.local_size() == 16 );
    maelstrom::set(vec, (uint32_t)rank);

    maelstrom::shuffle_to_rank(vec, 0);
    if(rank == 0) {
        assert(
            vec.local_size() == 16 * maelstrom::get_world_size()
        );

        size_t last = 0;
        for(size_t k = 0; k < vec.local_size(); ++k) {
            size_t current = std::any_cast<uint32_t>(vec.get_local(k));
            assert( current >= last );
            last = current;
        }
    }
}