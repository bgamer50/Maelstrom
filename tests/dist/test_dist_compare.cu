#include <sstream>

#include "test_utils.hpp"
#include "test_utils_dist.cuh"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/increment.h"
#include "maelstrom/algorithms/compare.h"
#include "maelstrom/algorithms/dist/shuffle.h"

using namespace maelstrom::test;

void test_dist_compare_basic(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    ncclComm_t comm;
    initialize_dist_env(argc, argv, &comm);

    try {
        for(auto storage : {maelstrom::DIST_HOST, maelstrom::DIST_DEVICE, maelstrom::DIST_MANAGED}) {
            test_dist_compare_basic(storage);
        }
    } catch(std::exception& err) {
                std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    teardown_dist_env();
    std::cout << "DONE!" << std::endl;
}

void test_dist_compare_basic(maelstrom::storage storage) {
    size_t rank = maelstrom::get_rank();
    size_t world_size = maelstrom::get_world_size();

    std::vector<float> A = {0.1f, 0.3f, 0.5f, 9.1f, 30.5f, 61.3f, -32.1f}; A.push_back(rank);
    std::vector<float> B = {0.2f, 0.5f, 0.5f, 9.9f, 30.5f, 61.6f, -33.1f}; B.push_back(rank);
    for(auto it = A.begin(); it != A.end(); ++it) *it = *it + rank/10.0;
    for(auto it = B.begin(); it != B.end(); ++it) *it = *it + rank/10.0;

    maelstrom::vector m_A(storage, maelstrom::float32, A.data(), A.size(), false);
    maelstrom::vector m_B(storage, maelstrom::float32, B.data(), B.size(), false);
    auto m_eq_cmp = maelstrom::compare(m_A, m_B, maelstrom::EQUALS);

    std::vector<bool> exp_eq = {false, false, true, false, true, false, false, true};
    assert( world_size * exp_eq.size() == m_eq_cmp.size() );
    for(size_t k = 0; k < m_A.local_size(); ++k) assert( (bool)std::any_cast<uint8_t>(m_eq_cmp.get_local(k)) ==  exp_eq[k] );
}