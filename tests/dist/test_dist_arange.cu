#include <sstream>

#include "test_utils.hpp"
#include "test_utils_dist.cuh"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/dist/shuffle.h"

using namespace maelstrom::test;

void test_dist_arange_basic();
void test_dist_arange_gap(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    ncclComm_t comm;
    initialize_dist_env(argc, argv, &comm);

    try {
        test_dist_arange_basic();
        for(auto storage : {maelstrom::DIST_HOST, maelstrom::DIST_DEVICE, maelstrom::DIST_MANAGED}) {
            test_dist_arange_gap(storage);
        }
    } catch(std::exception& err) {
                std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    teardown_dist_env();
    std::cout << "DONE!" << std::endl;
}

void test_dist_arange_basic() {
    size_t rank = maelstrom::get_rank();

    auto vec_0_20_1 = maelstrom::arange(
        maelstrom::DIST_DEVICE,
        static_cast<size_t>(0),
        static_cast<size_t>(21)
    );
    assert(vec_0_20_1.get_dtype() == maelstrom::uint64);

    for(size_t k = 0; k < vec_0_20_1.local_size(); ++k) std::cout << std::any_cast<size_t>(vec_0_20_1.get_local(k)) << " ";
    std::cout << std::endl;

    assert( vec_0_20_1.size() == 21 );
    size_t local_size = vec_0_20_1.local_size();

    maelstrom::vector zeros(maelstrom::DEVICE, maelstrom::uint64, local_size);
    maelstrom::set(zeros, static_cast<size_t>(0));
    maelstrom::shuffle(vec_0_20_1, zeros);

    auto expected_0_20_1 = maelstrom::arange(
        maelstrom::MANAGED,
        static_cast<size_t>(0),
        static_cast<size_t>(21)
    );

    if(rank == 0) {
        assert ( vec_0_20_1.local_size() == 21 );
        
        maelstrom::vector vec_0_20_1_copy(
            maelstrom::HOST,
            maelstrom::uint64,
            vec_0_20_1.data(),
            vec_0_20_1.local_size(), // reminder: size() will cause a hang here
            false
        );
        assert_array_equals(
            static_cast<size_t*>(vec_0_20_1_copy.data()),
            static_cast<size_t*>(expected_0_20_1.data()),
            21
        );
    }
}

void test_dist_arange_gap(maelstrom::storage storage) {
    size_t rank = maelstrom::get_rank();

    auto vec_4_1993_23 = maelstrom::arange(
        storage,
        static_cast<int32_t>(4),
        static_cast<int32_t>(1993),
        static_cast<int32_t>(23)
    );

    auto zeros = maelstrom::vector(
        maelstrom::single_storage_of(storage),
        maelstrom::uint64,
        vec_4_1993_23.local_size()
    );
    maelstrom::set(zeros, static_cast<size_t>(0));
    maelstrom::shuffle(vec_4_1993_23, zeros);

    if(rank == 0) {
        assert(
            vec_4_1993_23.local_size() == 87
        );
        auto expected = maelstrom::arange(
            maelstrom::HOST,
            static_cast<int32_t>(4),
            static_cast<int32_t>(1993),
            static_cast<int32_t>(23)
        );
        vec_4_1993_23 = vec_4_1993_23.to(maelstrom::HOST);

        assert_array_equals(
            static_cast<int32_t*>(vec_4_1993_23.data()),
            static_cast<int32_t*>(expected.data()),
            87
        );
    }
}