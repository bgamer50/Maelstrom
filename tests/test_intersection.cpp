#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/intersection.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_intersection_basic();
void test_intersection_duplicates();

int main(int argc, char* argv[]) {
    try {
        test_intersection_basic();
        test_intersection_duplicates();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_intersection_basic() {
    std::vector<float> cpp_array1 = {0.1f, 0.3f, 0.5f, 0.7f, -3.1f, 4.6f};

    maelstrom::vector m_array1(
        maelstrom::HOST,
        maelstrom::float32,
        cpp_array1.data(),
        cpp_array1.size(),
        true
    );

    std::vector<float> cpp_array2 = {0.1f, 0.7f, -3.1f, 6.2f};

    maelstrom::vector m_array2(
        maelstrom::HOST,
        maelstrom::float32,
        cpp_array2.data(),
        cpp_array2.size(),
        true
    );

    auto iix = maelstrom::intersection(m_array1, m_array2);
    assert( iix.size() == 3 );
    assert( iix.get_mem_type() == maelstrom::HOST );
    assert( iix.get_dtype() == maelstrom::uint64 );

    std::vector<size_t> expected = {(size_t)4, (size_t)0, (size_t)3};
    assert_array_equals(static_cast<size_t*>(iix.data()), expected.data(), 3);

    iix = maelstrom::intersection(m_array2, m_array1);
    expected = {(size_t)2, (size_t)0, (size_t)1};
    assert_array_equals(static_cast<size_t*>(iix.data()), expected.data(), 3);
}

void test_intersection_duplicates() {
    std::vector<float> cpp_array1 = {-6.6f, 0.1f, 0.1f, 0.3f, 0.5f, 0.7f, -3.1f, -6.6f, 4.6f};

    maelstrom::vector m_array1(
        maelstrom::MANAGED,
        maelstrom::float32,
        cpp_array1.data(),
        cpp_array1.size(),
        false
    );

    std::vector<float> cpp_array2 = {0.1f, 0.7f, -3.1f, -3.1f, 6.2f};

    maelstrom::vector m_array2(
        maelstrom::MANAGED,
        maelstrom::float32,
        cpp_array2.data(),
        cpp_array2.size(),
        false
    );

    auto iix = maelstrom::intersection(m_array1, m_array2);
    assert( iix.size() == 3 );
    assert( iix.get_mem_type() == maelstrom::MANAGED );
    assert( iix.get_dtype() == maelstrom::uint64 );

    std::vector<size_t> expected = {(size_t)6, (size_t)1, (size_t)5};
    assert_array_equals(static_cast<size_t*>(iix.data()), expected.data(), 3);

    iix = maelstrom::intersection(m_array2, m_array1);
    expected = {(size_t)2, (size_t)0, (size_t)1};
    assert_array_equals(static_cast<size_t*>(iix.data()), expected.data(), 3);
}