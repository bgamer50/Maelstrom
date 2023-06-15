#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/sort.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_sort_basic();

int main(int argc, char* argv[]) {
    try {
        test_sort_basic();
    } catch(std::exception& err) {
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
