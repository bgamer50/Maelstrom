#include <sstream>

#include "test_utils.hpp"
#include "test_utils_dist.cuh"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/increment.h"
#include "maelstrom/algorithms/count_unique.h"
#include "maelstrom/algorithms/dist/shuffle.h"
#include "maelstrom/algorithms/dist/bucket_sort.h"

using namespace maelstrom::test;

void test_dist_count_unique_sorted(maelstrom::storage storage);
void test_dist_count_unique_unsorted(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    ncclComm_t comm;
    initialize_dist_env(argc, argv, &comm);

    try {
        for(auto storage : {maelstrom::DIST_HOST, maelstrom::DIST_DEVICE, maelstrom::DIST_MANAGED}) {
            test_dist_count_unique_sorted(storage);
            test_dist_count_unique_unsorted(storage);
        }
    } catch(std::exception& err) {
                std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    teardown_dist_env();
    std::cout << "DONE!" << std::endl;
}

void test_dist_count_unique_sorted(maelstrom::storage storage) {
    size_t rank = maelstrom::get_rank();
    size_t world_size = maelstrom::get_world_size();

    std::vector<float> A = {0, 0, 0, 5, 7, 3, 3, 9, 14, 91};
    maelstrom::vector m_A(storage, maelstrom::float32, A.data(), A.size(), false);
    maelstrom::bucket_sort(m_A, maelstrom::UNIQUE);

    maelstrom::vector values;
    maelstrom::vector counts;
    
    std::tie(values, counts) = maelstrom::count_unique(m_A, 7, true);

    maelstrom::shuffle_to_rank(values, 0);
    maelstrom::shuffle_to_rank(counts, 0);
    values = values.to(maelstrom::HOST);
    counts = counts.to(maelstrom::HOST);
    if(rank == 0) {
        assert( values.local_size() == 7 );
        assert( counts.local_size() == 7 );

        std::vector<float> exp_values = {0, 3, 5, 7, 9, 14, 91};
        std::vector<size_t> exp_counts = {3L, 2L, 1L, 1L, 1L, 1L, 1L}; 
        for(size_t k = 0; k < exp_counts.size(); ++k) exp_counts[k] *= world_size;

        assert_array_equals(static_cast<float*>(values.data()), exp_values.data(), exp_values.size());
        assert_array_equals(static_cast<size_t*>(counts.data()), exp_counts.data(), exp_counts.size());
    }
}

void test_dist_count_unique_unsorted(maelstrom::storage storage) {

}