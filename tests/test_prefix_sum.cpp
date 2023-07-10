#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/prefix_sum.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_prefix_sum_basic();

int main(int argc, char* argv[]) {
    try {
        test_prefix_sum_basic();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_prefix_sum_basic() {
    std::vector<int> data = {1, 3, 0, 2, 4, 5, 0, 1, 2, 1, 2, 3, 4, 0, 2, 5};
    
    maelstrom::vector m_data(
        maelstrom::MANAGED,
        maelstrom::int32,
        data.data(),
        data.size(),
        false
    );

    maelstrom::prefix_sum(
        m_data
    );

    std::vector<int> expected = {1, 4, 4, 6, 10, 15, 15, 16, 18, 19, 21, 24, 28, 28, 30, 35};

    assert_array_equals(static_cast<int*>(m_data.data()), expected.data(), expected.size());
}