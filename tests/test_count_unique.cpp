#include "containers/vector.h"
#include "algorithms/sort.h"
#include "algorithms/count_unique.h"
#include "algorithms/select.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_count_unique_basic();

int main(int argc, char* argv[]) {
    try {
        test_count_unique_basic();
    } catch(std::exception& err) {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_count_unique_basic() {
    std::vector<double> cpp_array = {4.0, 6.3, 9.9, 6.3, 2.1, 2.1, 4.0, 4.0, 4.0, 6.3, 4.3, 4.2, 4.3};
    std::vector<double> expected_unique_values = {2.1, 4.0, 4.2, 4.3, 6.3, 9.9};
    std::vector<size_t> expected_unique_counts = {(size_t)2, (size_t)4, (size_t)1, (size_t)2, (size_t)3, (size_t)1};

    maelstrom::vector m_array(
        maelstrom::storage::MANAGED,
        maelstrom::float64,
        cpp_array.data(),
        cpp_array.size(),
        false
    );
    m_array.name = "m_array";

    maelstrom::vector unique_values;
    maelstrom::vector unique_counts;
    std::tie(unique_values, unique_counts) = maelstrom::count_unique(m_array);

    assert( unique_values.get_mem_type() == maelstrom::storage::MANAGED );
    assert( unique_values.get_dtype() == maelstrom::float64 );
    assert( unique_values.size() == 6 );
    assert_array_equals(static_cast<double*>(unique_values.data()), expected_unique_values.data(), expected_unique_values.size());

    assert( unique_counts.get_mem_type() == maelstrom::storage::MANAGED );
    assert( unique_counts.get_dtype() == maelstrom::uint64 );
    assert( unique_counts.size() == 6 );
    assert_array_equals(static_cast<size_t*>(unique_counts.data()), expected_unique_counts.data(), expected_unique_counts.size());
}
