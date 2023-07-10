#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/sort.h"
#include "maelstrom/algorithms/unique.h"
#include "maelstrom/algorithms/select.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_unique_basic();

int main(int argc, char* argv[]) {
    try {
        test_unique_basic();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_unique_basic() {
    std::vector<double> cpp_array = {4.0, 6.3, 9.9, 6.3, 2.1, 2.1, 4.0, 4.0, 4.0, 6.3, 4.3, 4.2, 4.3};

    maelstrom::vector m_array(
        maelstrom::storage::MANAGED,
        maelstrom::float64,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    auto unique = maelstrom::unique(m_array);

    assert( unique.get_dtype() == maelstrom::uint64 );
    unique = maelstrom::select(m_array, unique);
    std::vector<double> cpp_unique = {2.1, 4.0, 4.2, 4.3, 6.3, 9.9};
    assert( unique.get_dtype() == maelstrom::float64 );
    assert( unique.size() == 6 );
    assert_array_equals(static_cast<double*>(unique.data()), cpp_unique.data(), cpp_unique.size());

    maelstrom::sort(m_array);
    unique = maelstrom::unique(m_array, true);
    std::vector<uint64_t> cpp_unique_indices = {(size_t)0, (size_t)2, (size_t)6, (size_t)7, (size_t)9, (size_t)12};
    assert( unique.get_dtype() == maelstrom::uint64 );
    assert( unique.size() == 6 );
    assert_array_equals(static_cast<uint64_t*>(unique.data()), cpp_unique_indices.data(), cpp_unique_indices.size());
}
