#include "maelstrom/containers/vector.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_vector_basic();

int main(int argc, char* argv[]) {
    try {
        test_vector_basic();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_vector_basic() {
    std::vector<int> data = {0, 1, 2, 3, 4, 5, 6};

    maelstrom::vector m_data(
        maelstrom::storage::HOST,
        maelstrom::int32,
        data.data(),
        data.size(),
        false
    );

    std::vector<int> i_data = {10, 20, 30};
    maelstrom::vector m_i_data(
        maelstrom::storage::HOST,
        maelstrom::int32,
        i_data.data(),
        i_data.size(),
        true
    );

    m_i_data = m_i_data.to(maelstrom::storage::DEVICE);

    m_data.insert(
        3,
        m_i_data
    );

    int* m_data_ptr = static_cast<int*>(m_data.data());
    std::vector<int> correct_vals = {0, 1, 2, 10, 20, 30, 3, 4, 5, 6};
    assert_array_equals(m_data_ptr, correct_vals.data(), correct_vals.size());

    std::any val = m_data.get(3);
    assert( std::any_cast<int>(val) == 10);

    val = m_i_data.get(2);
    assert( std::any_cast<int>(val) == 30 );
}