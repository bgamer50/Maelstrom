#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/search_sorted.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_search_sorted_device();
void test_search_sorted_host();

int main(int argc, char* argv[]) {
    try {
        test_search_sorted_device();
        test_search_sorted_host();
    } catch(std::exception& err) {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_search_sorted_device() {
    std::vector<int> cpp_array = {0, 2, 5, 8, 11, 14};

    maelstrom::vector m_array(
        maelstrom::MANAGED,
        maelstrom::int32,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    std::vector<int> cpp_find = {8, 7, 13, 0, -1, 14, 13, 9, 12};

    maelstrom::vector m_find(
        maelstrom::MANAGED,
        maelstrom::int32,
        cpp_find.data(),
        cpp_find.size(),
        false
    );

    auto indices = maelstrom::search_sorted(m_array, m_find);
    assert( indices.size() == 9 );
    assert( indices.get_mem_type() == maelstrom::MANAGED );
    assert( indices.get_dtype() == maelstrom::uint64 );

    std::vector<size_t> expected = {(size_t)4, (size_t)3, (size_t)5, (size_t)1, (size_t)0, (size_t)6, (size_t)5, (size_t)4, (size_t)5};
    assert_array_equals(static_cast<size_t*>(indices.data()), expected.data(), 6);

}

void test_search_sorted_host() {
    std::vector<int> cpp_array = {0, 2, 5, 8, 11, 14};

    maelstrom::vector m_array(
        maelstrom::HOST,
        maelstrom::int32,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    std::vector<int> cpp_find = {8, 7, 13, 0, -1, 14, 13, 9, 12};

    maelstrom::vector m_find(
        maelstrom::HOST,
        maelstrom::int32,
        cpp_find.data(),
        cpp_find.size(),
        false
    );

    auto indices = maelstrom::search_sorted(m_array, m_find);
    assert( indices.size() == 9 );
    assert( indices.get_mem_type() == maelstrom::HOST );
    assert( indices.get_dtype() == maelstrom::uint64 );

    std::vector<size_t> expected = {(size_t)4, (size_t)3, (size_t)5, (size_t)1, (size_t)0, (size_t)6, (size_t)5, (size_t)4, (size_t)5};
    assert_array_equals(static_cast<size_t*>(indices.data()), expected.data(), 6);

}