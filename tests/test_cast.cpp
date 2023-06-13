#include "containers/vector.h"
#include "algorithms/compare.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_cast_basic();

int main(int argc, char* argv[]) {
    try {
        test_cast_basic();
    } catch(std::exception& err) {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_cast_basic() {
    std::vector<int> cpp_array = {1, 3, 5, 7, 9, 11, 13, 15};

    maelstrom::vector m_array(
        maelstrom::storage::MANAGED,
        maelstrom::int32,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    auto m_array_float32 = m_array.astype(maelstrom::float32);
    assert( m_array_float32.get_mem_type() == maelstrom::storage::MANAGED );
    assert( m_array_float32.get_dtype() == maelstrom::float32 );
    assert( m_array_float32.size() == 8 );

    std::vector<float> float_array = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f};
    assert_array_equals(static_cast<float*>(m_array_float32.data()), float_array.data(), 8);

    std::vector<int8_t> cpp_array8 = {(int8_t)1, (int8_t)3, (int8_t)5, (int8_t)7, (int8_t)9, (int8_t)11, (int8_t)13, (int8_t)15};

    maelstrom::vector m_array8(
        maelstrom::storage::MANAGED,
        maelstrom::int8,
        cpp_array8.data(),
        cpp_array8.size(),
        false
    );

    auto m_array_float64 = m_array8.astype(maelstrom::float64);
    assert( m_array_float64.get_mem_type() == maelstrom::storage::MANAGED );
    assert( m_array_float64.get_dtype() == maelstrom::float64 );
    assert( m_array_float64.size() == 8 );

    std::vector<double> double_array = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0};
    assert_array_equals(static_cast<double*>(m_array_float64.data()), double_array.data(), 8);
}