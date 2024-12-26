#include <sstream>

#include "test_utils.hpp"
#include "test_utils_dist.cuh"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/assign.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/dist/shuffle.h"

using namespace maelstrom::test;

void test_dist_assign_basic(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    ncclComm_t comm;
    initialize_dist_env(argc, argv, &comm);

    try {
        for(auto storage : {maelstrom::DIST_HOST, maelstrom::DIST_DEVICE, maelstrom::DIST_MANAGED}) {
            test_dist_assign_basic(storage);
        }
    } catch(std::exception& err) {
                std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    teardown_dist_env();
    std::cout << "DONE!" << std::endl;
}

void test_dist_assign_basic(maelstrom::storage storage) {
    size_t rank = maelstrom::get_rank();

    auto vec = maelstrom::arange(storage, (uint8_t)100);
    auto ix = maelstrom::arange(storage, (size_t)1, (size_t)100, (size_t)2);
    auto val = maelstrom::arange(storage, (uint8_t)10, (uint8_t)110, (uint8_t)2);

    maelstrom::assign(vec, ix, val);
    maelstrom::shuffle_to_rank(vec, 0);

    if(rank == 0) {
        vec = vec.to(maelstrom::HOST);

        for(size_t k = 0; k < vec.size(); ++k) {
            if(k % 2 == 0) assert( std::any_cast<uint8_t>(vec.get(k)) ==  k);
            else assert( std::any_cast<uint8_t>(vec.get(k)) == (k + 9) );
        }
    }
}