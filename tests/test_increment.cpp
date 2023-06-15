#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/increment.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_increment_basic();
void test_increment_start_end();

int main(int argc, char* argv[]) {
    try {
        test_increment_basic();
        test_increment_start_end();
    } catch(std::exception& err) {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_increment_basic() {
    std::vector<int> cpp_array = {0, 1, 2, 3, 4, 5, 6, 7};

    maelstrom::vector m_array(
        maelstrom::storage::DEVICE,
        maelstrom::int32,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    maelstrom::increment(
        m_array,
        (int32_t)10
    );
    m_array = m_array.to(maelstrom::storage::MANAGED);

    std::vector<int> correct_array_incremented = {10, 11, 12, 13, 14, 15, 16, 17};
    assert_array_equals(
        static_cast<int*>(m_array.data()),
        correct_array_incremented.data(),
        m_array.size()
    );

    maelstrom::increment(
        m_array,
        (int)27
    );

    std::vector<int> correct_array_implemented_2 = {37, 38, 39, 40, 41, 42, 43, 44};
    assert_array_equals(
        static_cast<int*>(m_array.data()),
        correct_array_implemented_2.data(),
        m_array.size()
    );
}

void test_increment_start_end() {
    std::vector<float> array = {0.0f, 3.4f, 5.1f, 2.345f, 3.999f};
    maelstrom::vector m_array(
        maelstrom::storage::HOST,
        maelstrom::float32,
        array.data(),
        array.size(),
        true
    );

    maelstrom::increment(
        m_array,
        (float)0.1
    );

    std::vector<float> expected_result = {0.1, 3.5, 5.2, 2.445, 4.099};
    assert_vector_equals(array, expected_result);
}