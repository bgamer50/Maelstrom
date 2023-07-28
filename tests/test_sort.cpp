#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/sort.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_sort_basic();
void test_sort_multi();
void test_sort_mixed_int();

int main(int argc, char* argv[]) {
    try {
        test_sort_basic();
        test_sort_multi();
        test_sort_mixed_int();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_sort_basic() {
    std::vector<double> cpp_array = {9.1, 8.2, 4.3, 2.2, 4.5, 9.81};

    maelstrom::vector m_array(
        maelstrom::storage::MANAGED,
        maelstrom::float64,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    auto sorted_ix = maelstrom::sort(m_array);
    
    std::vector<double> cpp_sorted_values = {2.2, 4.3, 4.5, 8.2, 9.1, 9.81};
    assert( m_array.size() == cpp_sorted_values.size() );
    assert( m_array.get_dtype() == maelstrom::float64 );
    assert_array_equals(static_cast<double*>(m_array.data()), cpp_sorted_values.data(), cpp_sorted_values.size());

    std::vector<size_t> cpp_sorted_ix = {(size_t)3, (size_t)2, (size_t)4, (size_t)1, (size_t)0, (size_t)5};
    assert( sorted_ix.size() == cpp_sorted_ix.size() );
    assert( sorted_ix.get_dtype() == maelstrom::uint64 );
    assert_array_equals(static_cast<size_t*>(sorted_ix.data()), cpp_sorted_ix.data(), cpp_sorted_ix.size());
}

void test_sort_multi() {
    std::vector<double> a0 = {1.1, 2.1, 3.1, 1.1, 2.1, 2.1, 3.1, 2.1};
    std::vector<double> a1 = {5.5, 6.5, 7.5, 8.5, 9.5, 3.2, 7.5, 3.2};
    std::vector<double> a2 = {3.3, 3.3, 3.5, 4.3, 4.5, 1.2, 3.5, 8.6};
    std::vector<double> a3 = {1.1, 2.1, 3.1, 4.1, 5.6, 6.1, 2.1, 4.1};

    maelstrom::vector m_array_0(
        maelstrom::storage::MANAGED,
        maelstrom::float64,
        a0.data(),
        a0.size(),
        false
    );
    maelstrom::vector m_array_1(
        maelstrom::storage::MANAGED,
        maelstrom::float64,
        a1.data(),
        a1.size(),
        false
    );
    maelstrom::vector m_array_2(
        maelstrom::storage::MANAGED,
        maelstrom::float64,
        a2.data(),
        a2.size(),
        false
    );
    maelstrom::vector m_array_3(
        maelstrom::storage::MANAGED,
        maelstrom::float64,
        a3.data(),
        a3.size(),
        false
    );

    auto ix = maelstrom::sort({
        std::ref(m_array_0),
        std::ref(m_array_1),
        std::ref(m_array_2),
        std::ref(m_array_3)
    });

    std::vector<double> expected_0 = {1.1, 1.1, 2.1, 2.1, 2.1, 2.1, 3.1, 3.1};
    std::vector<double> expected_1 = {5.5, 8.5, 3.2, 3.2, 6.5, 9.5, 7.5, 7.5};
    std::vector<double> expected_2 = {3.3, 4.3, 1.2, 8.6, 3.3, 4.5, 3.5, 3.5};
    std::vector<double> expected_3 = {1.1, 4.1, 6.1, 4.1, 2.1, 5.6, 2.1, 3.1};

    assert_array_equals(static_cast<double*>(m_array_0.data()), expected_0.data(), expected_0.size());
    assert_array_equals(static_cast<double*>(m_array_1.data()), expected_1.data(), expected_1.size());
    assert_array_equals(static_cast<double*>(m_array_2.data()), expected_2.data(), expected_2.size());
    assert_array_equals(static_cast<double*>(m_array_3.data()), expected_3.data(), expected_3.size());
}

void test_sort_mixed_int() {
    std::vector<uint64_t> vec1 = {5ul, 0ul, 7ul, 6ul, 3ul, 2ul, 0ul, 9ul, 6ul, 6ul, 7ul};
    std::vector<uint32_t> vec2 = {5u , 5u , 5u , 1u , 2u , 3u , 5u , 4u , 3u , 2u , 1u };
    std::vector<uint64_t> vec3 = {7ul, 6ul, 5ul, 5ul, 6ul, 8ul, 7ul, 2ul, 3ul, 1ul, 7ul};

    maelstrom::vector m_vec1(maelstrom::MANAGED, maelstrom::uint64, vec1.data(), vec1.size(), false);
    maelstrom::vector m_vec2(maelstrom::MANAGED, maelstrom::uint32, vec2.data(), vec2.size(), false);
    maelstrom::vector m_vec3(maelstrom::MANAGED, maelstrom::uint64, vec3.data(), vec3.size(), false);

    auto ix = maelstrom::sort({std::ref(m_vec1), std::ref(m_vec2), std::ref(m_vec3)});

    std::vector<uint64_t> expected_1 = {0, 0, 2, 3, 5, 6, 6, 6, 7, 7, 9};
    std::vector<uint32_t> expected_2 = {5, 5, 3, 2, 5, 1, 2, 3, 1, 5, 4};
    std::vector<uint64_t> expected_3 = {6, 7, 8, 6, 7, 5, 1, 3, 7, 5, 2};

    assert_array_equals(static_cast<uint64_t*>(m_vec1.data()), expected_1.data(), expected_1.size());
    assert_array_equals(static_cast<uint32_t*>(m_vec2.data()), expected_2.data(), expected_2.size());
    assert_array_equals(static_cast<uint64_t*>(m_vec3.data()), expected_3.data(), expected_3.size());
}
