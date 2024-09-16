#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/similarity.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_similarity_basic_cosine();
void test_similarity_multi_cosine();

int main(int argc, char* argv[]) {
    try {
        test_similarity_basic_cosine();
        test_similarity_multi_cosine();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_similarity_basic_cosine() {
    std::vector<double> cpp_array_A = {9.1, 8.2, 4.3, 2.2, 4.5, 9.81};
    std::vector<double> cpp_array_B = {9.1, 3.2, 4.1, -6.9, 7.4, 10.11};

    maelstrom::vector m_A(
        maelstrom::PINNED,
        maelstrom::float64,
        cpp_array_A.data(),
        cpp_array_A.size(),
        false
    );

    maelstrom::vector m_B(
        maelstrom::PINNED,
        maelstrom::float64,
        cpp_array_B.data(),
        cpp_array_B.size(),
        false
    );

    maelstrom::vector offsets;
    auto sim = maelstrom::similarity(
        maelstrom::COSINE,
        m_A,
        offsets,
        m_B,
        6
    );

    assert( sim.size() == 1 );
    assert( sim.get_dtype() == maelstrom::float64 );
    assert( sim.get_mem_type() == maelstrom::PINNED );
    assert( std::any_cast<double>(sim.get(0)) - 0.808134 < 0.000002 );
}

void test_similarity_multi_cosine() {
    std::vector<double> cpp_emb_A = {
        9.1, 8.2, 4.3, 2.2, 4.5, 9.81,
        3.3, 2.2, 1.1, 7.56, -7.7, -10.5,
        5.0, 5.0, 6.3, 6.3, 1.0, 1.2,
        -1.0, -1.0, 3.7, 6.7, 9.7, -3.1,
        -9.1, -9.3, -9.5, -9.7, 0.0, 1.0,
        5.5, 11.1, 3.1, -1.5, 4.0, 5.0
    };
    
    std::vector<double> cpp_emb_B = {
        9.1, 3.2, 4.1, -6.9, 7.4, 10.11,
        5.1, 6.1, 3.1, 5.1, -4.1, -7.1,
        -3.3, -4.3, 1.0, 4.0, 8.9, 9.9
    };

    maelstrom::vector m_A(
        maelstrom::PINNED,
        maelstrom::float64,
        cpp_emb_A.data(),
        cpp_emb_A.size(),
        false
    );

    maelstrom::vector m_B(
        maelstrom::PINNED,
        maelstrom::float64,
        cpp_emb_B.data(),
        cpp_emb_B.size(),
        false
    );

    maelstrom::vector offsets;
    auto sim = maelstrom::similarity(
        maelstrom::COSINE,
        m_A,
        offsets,
        m_B,
        6
    );

    assert( sim.size() == 6 );
    assert( sim.get_dtype() == maelstrom::float64 );
    assert( sim.get_mem_type() == maelstrom::PINNED );

    std::vector<double> expected_sim = {0.808134, 0.886621, 0.642367, 0.489099, 0.112288, 0.741165};

    for(size_t k = 0; k < sim.size(); ++k) {
        assert(
            std::any_cast<double>(sim.get(k)) - expected_sim[k] < 0.000002
        );
    }
    
}