#include <sstream>

#include "test_utils.hpp"
#include "test_utils_dist.cuh"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/dist/shuffle.h"

using namespace maelstrom::test;

void test_dist_shuffle_basic(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    ncclComm_t comm;
    initialize_dist_env(argc, argv, &comm);

    try {
        for(auto storage : {maelstrom::DIST_HOST, maelstrom::DIST_DEVICE, maelstrom::DIST_MANAGED}) {
            test_dist_shuffle_basic(storage);
        }
    } catch(std::exception& err) {
                std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    teardown_dist_env();
    std::cout << "DONE!" << std::endl;
}

void test_dist_shuffle_basic(maelstrom::storage storage) {
    size_t rank = maelstrom::get_rank();

    auto vec = maelstrom::arange(
        storage,
        static_cast<int64_t>(11),
        static_cast<int64_t>(65588)
    );
    assert(vec.get_dtype() == maelstrom::int64);

    assert( vec.size() == 65577 );
    size_t local_size = vec.local_size();

    maelstrom::vector zeros(maelstrom::DEVICE, maelstrom::uint64, local_size);
    maelstrom::set(zeros, static_cast<size_t>(0));
    maelstrom::shuffle(vec, zeros);

    if(rank == 0) {
        auto expected = maelstrom::arange(maelstrom::HOST, (int64_t)11, (int64_t)65588);

        assert ( vec.local_size() == 65577 );
        
        maelstrom::vector vec_local = vec.to(maelstrom::HOST);
        assert( vec_local.get_dtype() == maelstrom::int64 );
        assert_array_equals(
            static_cast<size_t*>(vec_local.data()),
            static_cast<size_t*>(expected.data()),
            65577
        );
    } else {
        assert( vec.local_size() == 0 );
    }

    assert(vec.size() == 65577);
}