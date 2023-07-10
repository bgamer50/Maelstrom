#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/remove_if.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_remove_basic();

int main(int argc, char* argv[]) {
    try {
        test_remove_basic();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_remove_basic() {
    std::vector<int> cpp_array = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<char> cpp_stencil = {(char)0, (char)1, (char)0, (char)1, (char)0, char(1), char(0), char(2) };

    assert ( cpp_array.size() == cpp_stencil.size() );

    maelstrom::vector m_array(
        maelstrom::storage::DEVICE,
        maelstrom::int32,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    maelstrom::vector m_stencil(
        maelstrom::storage::DEVICE,
        maelstrom::int8,
        cpp_stencil.data(),
        cpp_stencil.size(),
        false
    );

    maelstrom::remove_if(m_array, m_stencil);
    assert ( m_array.size() == 4 );
    m_array = m_array.to(maelstrom::storage::HOST);

    std::vector<int> expected_result = {0, 2, 4, 6};
    assert_array_equals(expected_result.data(), static_cast<int*>(m_array.data()), 4);
}