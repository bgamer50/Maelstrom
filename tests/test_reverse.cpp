#include "maelstrom/algorithms/reverse.h"

#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_reverse_basic();

int main(int argc, char* argv[]) {
    try {
        test_reverse_basic();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_reverse_basic() {
    std::vector<char> cpp_array = {(char)0, (char)1, (char)0, (char)1, (char)0, char(1), char(0), char(2) };

    maelstrom::vector m_array(
        maelstrom::storage::DEVICE,
        maelstrom::int8,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    maelstrom::reverse(m_array);
    assert ( m_array.size() == cpp_array.size() );
    m_array = m_array.to(maelstrom::storage::HOST);

    std::reverse(cpp_array.begin(), cpp_array.end());
    assert_array_equals(cpp_array.data(), static_cast<char*>(m_array.data()), cpp_array.size());
}