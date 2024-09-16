#include <sstream>

#include "test_utils.hpp"
#include "test_utils_dist.cuh"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/increment.h"
#include "maelstrom/algorithms/sort.h"
#include "maelstrom/algorithms/dist/bucket_sort.h"
#include "maelstrom/algorithms/dist/shuffle.h"

using namespace maelstrom::test;

void test_dist_sort_basic(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    ncclComm_t comm;
    initialize_dist_env(argc, argv, &comm);

    try {
        for(auto storage : {maelstrom::DIST_DEVICE, maelstrom::DIST_HOST, maelstrom::DIST_MANAGED}) {
            test_dist_sort_basic(storage);
        }
    } catch(std::exception& err) {
                std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    teardown_dist_env();
    std::cout << "DONE!" << std::endl;
}

void test_dist_sort_basic(maelstrom::storage storage) {
    size_t rank = maelstrom::get_rank();
    size_t world_size = maelstrom::get_world_size();

    std::vector<float> A = {0.5f, 0.1f, 0.3f, 9.1f, 30.5f, 0.7f, 61.3f, -32.1f, 0.7f};
    for(size_t k = 0; k < 2*rank; ++k) A.push_back((float)rank);

    maelstrom::vector m_A(maelstrom::single_storage_of(storage), maelstrom::float32, A.data(), A.size(), false);
    m_A = maelstrom::to_dist_vector(std::move(m_A));

    std::vector<float> exp_sort = {-32.1f, 0.1f, 0.3f, 0.5f, 0.7f, 0.7f, 9.1f, 30.5f, 51.3f};
    
    auto ix = maelstrom::bucket_sort(m_A, maelstrom::ORIGINAL);

    // partition sizes should be the same
    assert( m_A.local_size() == A.size() );

    assert( ix.size() == m_A.size() );
    assert( ix.local_size() == m_A.local_size() );

    maelstrom::shuffle_to_rank(m_A, 0);
    if(rank == 0) {
        for(size_t k = 1; k < m_A.local_size(); ++k) {
            assert( std::any_cast<float>(m_A.get_local(k)) >= std::any_cast<float>(m_A.get_local(k - 1)) );
        }
    }

}