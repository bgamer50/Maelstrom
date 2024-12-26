#include <sstream>

#include "test_utils.hpp"
#include "test_utils_dist.cuh"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/increment.h"
#include "maelstrom/algorithms/cast.h"
#include "maelstrom/algorithms/dist/shuffle.h"

using namespace maelstrom::test;

void test_dist_cast_basic(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    ncclComm_t comm;
    initialize_dist_env(argc, argv, &comm);

    try {
        for(auto storage : {maelstrom::DIST_HOST, maelstrom::DIST_DEVICE, maelstrom::DIST_MANAGED}) {
            test_dist_cast_basic(storage);
        }
    } catch(std::exception& err) {
                std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    teardown_dist_env();
    std::cout << "DONE!" << std::endl;
}

void test_dist_cast_basic(maelstrom::storage storage) {
    size_t rank = maelstrom::get_rank();
    size_t world_size = maelstrom::get_world_size();

    std::vector<int> vec_host = {0, 10, 15, 9};
    for(auto it = vec_host.begin(); it != vec_host.end(); ++it) *it = *it + rank;

    auto m_vec = maelstrom::vector(
        storage,
        maelstrom::uint32,
        vec_host.data(),
        vec_host.size(),
        false
    );

    m_vec = maelstrom::cast(m_vec, maelstrom::uint8);
    maelstrom::shuffle_to_rank(m_vec, 0);

    if(rank == 0) {
        auto vec = m_vec.to(maelstrom::HOST);
        assert( vec.size() == vec_host.size() * world_size );

        for(size_t k = 0; k < world_size; ++k) {
            auto ld = static_cast<uint8_t*>(vec.data()) + (vec_host.size() * k);
            for(size_t i = 0; i < 4; ++i) {
                assert( ld[i] == (vec_host[i] + k));
            }
        }
    }
}