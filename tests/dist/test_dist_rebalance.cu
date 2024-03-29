#include <sstream>

#include "test_utils.hpp"
#include "test_utils_dist.cuh"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/reduce.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/dist/rebalance.h"

using namespace maelstrom::test;

void test_dist_rebalance_basic(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    ncclComm_t comm;
    initialize_dist_env(argc, argv, &comm);

    try {
        for(auto storage : {maelstrom::DIST_HOST, maelstrom::DIST_DEVICE, maelstrom::DIST_MANAGED}) {
            test_dist_rebalance_basic(storage);
        }
    } catch(std::exception& err) {
                std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    teardown_dist_env();
    std::cout << "DONE!" << std::endl;
}

void test_dist_rebalance_basic(maelstrom::storage storage) {
    size_t rank = maelstrom::get_rank();

    maelstrom::vector vec(storage, maelstrom::uint32);
    vec.resize_local((rank + 1) * 2);
    assert( vec.local_size() == (rank + 1) * 2 );
    maelstrom::set(vec, (uint32_t)rank);

    auto r = arange(maelstrom::HOST, 1, (int)maelstrom::get_world_size() + 1);
    std::any sum;
    std::tie(sum, std::ignore) = maelstrom::reduce(r, maelstrom::SUM);
    size_t expected_size = std::any_cast<int>(sum) * 2;

    assert( vec.size() == expected_size );

    maelstrom::rebalance(vec);
    assert( vec.size() == expected_size );

    uint32_t last = 0;
    for(size_t k = 0; k < vec.local_size(); ++k) {
        uint32_t current = std::any_cast<uint32_t>(vec.get_local(k));
        assert( current >= last );
        last = current;
    }
}