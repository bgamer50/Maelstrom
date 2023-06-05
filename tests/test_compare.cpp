#include "containers/vector.h"
#include "algorithms/compare.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_compare_basic();
void test_compare_select();

int main(int argc, char* argv[]) {
    try {
        test_compare_basic();
        test_compare_select();
    } catch(std::exception& err) {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_compare_basic() {
    std::vector<int> cpp_array1 = {1, 3, 5, 7, 9, 11, 13, 15};
    std::vector<int> cpp_array2 = {1, 3, 5, 7, 2, 4, 6, 8};

    maelstrom::vector m_array1(
        maelstrom::storage::MANAGED,
        maelstrom::int32,
        cpp_array1.data(),
        cpp_array1.size(),
        false
    );

    maelstrom::vector m_array2(
        maelstrom::storage::MANAGED,
        maelstrom::int32,
        cpp_array2.data(),
        cpp_array2.size(),
        false
    );

    auto output_vec = maelstrom::compare(
        m_array1,
        m_array2,
        maelstrom::comparator::EQUALS
    );

    assert( output_vec.size() == m_array1.size() );
    assert( output_vec.get_dtype() == maelstrom::uint8 );
    assert( output_vec.get_mem_type() == maelstrom::storage::MANAGED );

    std::vector<uint8_t> correct_results = {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0, (uint8_t)0};
    assert_array_equals(static_cast<uint8_t*>(output_vec.data()), correct_results.data(), correct_results.size());

    output_vec = maelstrom::compare(m_array1, m_array1, maelstrom::comparator::NOT_EQUALS);
    correct_results = {(uint8_t)0, (uint8_t)0, (uint8_t)0, (uint8_t)0, (uint8_t)0, (uint8_t)0, (uint8_t)0, (uint8_t)0};
    assert( output_vec.size() == 8 );
    assert_array_equals(static_cast<uint8_t*>(output_vec.data()), correct_results.data(), 8);

    output_vec = maelstrom::compare(m_array1, m_array2, maelstrom::comparator::GREATER_THAN);
    correct_results = {(uint8_t)0, (uint8_t)0, (uint8_t)0, (uint8_t)0, (uint8_t)1, (uint8_t)1, (uint8_t)1,(uint8_t)1};
    assert( output_vec.size() == 8 );
    assert_array_equals(static_cast<uint8_t*>(output_vec.data()), correct_results.data(), 8);

    // Now change things a bit
    cpp_array1 = {-1, 3, 0, -4, 2, 5, 8, -1, -2, -1};
    cpp_array2 = {0, 4, 1, 0, 3, 3, -5, -6, 8, 2};

    m_array1 = maelstrom::vector(
        maelstrom::storage::HOST,
        maelstrom::int32,
        cpp_array1.data(),
        cpp_array1.size(),
        false
    );

    m_array2 = maelstrom::vector(
        maelstrom::storage::HOST,
        maelstrom::int32,
        cpp_array2.data(),
        cpp_array2.size(),
        false
    );

    output_vec = maelstrom::compare(m_array1, m_array2, maelstrom::comparator::LESS_THAN_OR_EQUAL);

    correct_results = {(uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)0, (uint8_t)1, (uint8_t)1};
    assert( output_vec.size() == 10 );
    assert_array_equals(static_cast<uint8_t*>(output_vec.data()), correct_results.data(), 10);
}

void test_compare_select() {
    std::vector<float> cpp_array1 = {-0.3f, 4.54f, 2.1f, 4.2f, 3.54f, 3.54f, -0.3f};
    std::vector<float> cpp_array2 = {-0.3f, 0.00f, 0.2f, -3.2f, 3.54f, 3.54f, 3.54f};

    auto m_array1 = maelstrom::vector(
        maelstrom::storage::HOST,
        maelstrom::float32,
        cpp_array1.data(),
        cpp_array1.size(),
        false
    );

    auto m_array2 = maelstrom::vector(
        maelstrom::storage::HOST,
        maelstrom::float32,
        cpp_array2.data(),
        cpp_array2.size(),
        false
    );

    auto output_vec = maelstrom::compare_select(m_array1, m_array2, maelstrom::comparator::NOT_EQUALS);

    std::vector<float> correct_results = {4.54f, 2.1f, 4.2f, -0.3f};
    assert( output_vec.size() == 4 );
    assert( output_vec.get_dtype() == maelstrom::float32 );
    assert( output_vec.get_mem_type() == maelstrom::storage::HOST );

    assert_array_equals(static_cast<float*>(output_vec.data()), correct_results.data(), 4);
}