#include "containers/vector.h"
#include "algorithms/set.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_set_basic();

int main(int argc, char* argv[]) {
    try {
        test_set_basic();
    } catch(std::exception& err) {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_set_basic() {
    std::vector<double> cpp_array = {9.1, 8.2, 4.3, 2.2, 4.5, 9.81};

    maelstrom::vector m_array(
        maelstrom::storage::MANAGED,
        maelstrom::float64,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    maelstrom::set(
        m_array,
        0,
        4,
        62.0
    );
    m_array = m_array.to(maelstrom::storage::HOST);
    
    std::vector<double> expected_result = {62.0, 62.0, 62.0, 62.0, 4.5, 9.81};
    assert_array_equals(
        static_cast<double*>(m_array.data()),
        expected_result.data(),
        expected_result.size()
    );

    maelstrom::set(
        m_array,
        5,
        6,
        11.9
    );

    std::vector<double> expected_result_2 = {62.0, 62.0, 62.0, 62.0, 4.5, 11.9};
    assert_array_equals(
        static_cast<double*>(m_array.data()),
        expected_result_2.data(),
        expected_result_2.size()
    );
}
