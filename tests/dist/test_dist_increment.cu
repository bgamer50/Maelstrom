#include <sstream>

#include "test_utils.hpp"
#include "test_utils_dist.cuh"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/increment.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/dist/shuffle.h"

using namespace maelstrom::test;

void test_dist_increment_basic(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    ncclComm_t comm;
    initialize_dist_env(argc, argv, &comm);

    try {
        for(auto storage : {maelstrom::DIST_HOST, maelstrom::DIST_DEVICE, maelstrom::DIST_MANAGED}) {
            test_dist_increment_basic(storage);
        }
    } catch(std::exception& err) {
                std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    teardown_dist_env();
    std::cout << "DONE!" << std::endl;
}

void test_dist_increment_basic(maelstrom::storage storage) {
    size_t rank = maelstrom::get_rank();
    uint32_t inc = 49;

    auto vec = maelstrom::arange(storage, (uint32_t)100);
    auto exp = maelstrom::arange(maelstrom::HOST, inc, inc + 100);

    maelstrom::increment(vec, inc);
    maelstrom::shuffle_to_rank(vec, 0);

    if(rank == 0) {
        vec = vec.to(maelstrom::HOST);
        assert( vec.size() == exp.size() );
        assert_array_equals<uint32_t>(
            static_cast<uint32_t*>(vec.data()),
            static_cast<uint32_t*>(exp.data()),
            vec.size()
        );
    }
}