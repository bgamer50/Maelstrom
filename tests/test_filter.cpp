#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/filter.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_filter_basic();
void test_filter_advanced();

int main(int argc, char* argv[]) {
    try {
        test_filter_basic();
        test_filter_advanced();
    } catch(std::exception& err) {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_filter_basic() {
    std::vector<int> cpp_array = {-10, 0, 1, 2, 3, 4, 5, 6, 7, 9};

    maelstrom::vector m_array(
        maelstrom::storage::MANAGED,
        maelstrom::int32,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    auto output_arr = maelstrom::filter(
        m_array,
        maelstrom::comparator::EQUALS,
        static_cast<int32_t>(32)
    );

    assert( output_arr.get_mem_type() == maelstrom::storage::MANAGED );
    assert( output_arr.get_dtype() == maelstrom::uint64 );
    assert( output_arr.size() == 0 );

    output_arr = maelstrom::filter(
        m_array,
        maelstrom::comparator::LESS_THAN,
        static_cast<int32_t>(7)
    );

    assert( output_arr.get_mem_type() == maelstrom::storage::MANAGED );
    assert( output_arr.get_dtype() == maelstrom::uint64 );
    assert( output_arr.size() == 8 );

    std::vector<size_t> correct_result = {(size_t)0, (size_t)1, (size_t)2, (size_t)3, (size_t)4, (size_t)5, (size_t)6, (size_t)7};
    assert_array_equals(correct_result.data(), static_cast<size_t*>(output_arr.data()), 8);
}

void test_filter_advanced() {
    std::vector<int8_t> cpp_array = {(int8_t)2, (int8_t)0, (int8_t)-49, (int8_t)3, (int8_t)8, (int8_t)2, (int8_t)-23, (int8_t)1, (int8_t)32, (int8_t)34};

    maelstrom::vector m_array(
        maelstrom::storage::MANAGED,
        maelstrom::int8,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    auto output_vec = maelstrom::filter(
        m_array,
        maelstrom::comparator::EQUALS,
        (int8_t)2
    );

    assert( output_vec.size() == 2 );
    std::vector<size_t> correct_result = {(size_t)0, (size_t)5};
    assert_array_equals(correct_result.data(), static_cast<size_t*>(output_vec.data()), 2);

    output_vec = maelstrom::filter(
        m_array,
        maelstrom::comparator::LESS_THAN_OR_EQUAL,
        (int8_t)3
    );
    correct_result = {(size_t)0, (size_t)1, (size_t)2, (size_t)3, (size_t)5, (size_t)6, (size_t)7};
    assert( output_vec.size() ==  correct_result.size());
    assert_array_equals(correct_result.data(), static_cast<size_t*>(output_vec.data()), correct_result.size());
}
