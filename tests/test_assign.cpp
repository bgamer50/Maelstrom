#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/assign.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_assign_basic();

int main(int argc, char* argv[]) {
    try {
        test_assign_basic();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_assign_basic() {
    std::vector<float> cpp_array = {0.01f, 0.03f, 0.05f, 0.07f, 0.09f, 0.11f, 0.13f, 0.15f};

    maelstrom::vector m_array(
        maelstrom::storage::MANAGED,
        maelstrom::float32,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    std::vector<uint64_t> cpp_index = {4, 5, 0};
    std::vector<float> cpp_index_vals = {1.5f, 1.6f, 1.7f};

    maelstrom::vector m_index(
        maelstrom::storage::MANAGED,
        maelstrom::int64,
        cpp_index.data(),
        cpp_index.size(),
        false
    );

    maelstrom::vector m_index_vals(
        maelstrom::storage::DEVICE,
        maelstrom::float32,
        cpp_index_vals.data(),
        cpp_index_vals.size(),
        false
    );

    maelstrom::assign(
        m_array,
        m_index,
        m_index_vals
    );

    std::vector<float> expected = {1.7f, 0.03f, 0.05f, 0.07f, 1.5f, 1.6f, 0.13f, 0.15f};
    assert_array_equals(static_cast<float*>(m_array.data()), expected.data(), expected.size());

}