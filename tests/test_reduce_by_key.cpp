#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/reduce_by_key.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_reduce_basic();
void test_reduce_mean();

int main(int argc, char* argv[]) {
    try {
        test_reduce_basic();
        test_reduce_mean();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_reduce_basic() {
    std::vector<uint8_t> cpp_array = {(uint8_t)2, (uint8_t)5, (uint8_t)3, (uint8_t)16, (uint8_t)4, (uint8_t)4, (uint8_t)11, (uint8_t)17};
    std::vector<int64_t> cpp_keys =  {(int64_t)2, (int64_t)5, (int64_t)9, (int64_t)13, (int64_t)2, (int64_t)13, (int64_t)2, (int64_t)5};

    maelstrom::vector m_array(
        maelstrom::storage::DEVICE,
        maelstrom::uint8,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    maelstrom::vector m_keys(
        maelstrom::storage::DEVICE,
        maelstrom::int64,
        cpp_keys.data(),
        cpp_keys.size()
    );

    maelstrom::vector output_values;
    maelstrom::vector output_indices;

    // Testing SUM
    std::tie(output_values, output_indices) = maelstrom::reduce_by_key(
        m_keys,
        m_array,
        maelstrom::SUM
    );

    assert( output_values.get_mem_type() == maelstrom::storage::DEVICE );
    assert( output_indices.get_mem_type() == maelstrom::storage::DEVICE );

    assert( output_values.get_dtype() == maelstrom::uint8 );
    assert( output_indices.get_dtype() == maelstrom::uint64 );

    assert( output_values.size() == 4 );
    assert( output_indices.size() == 4 );

    output_values = output_values.to(maelstrom::storage::MANAGED);
    output_indices = output_indices.to(maelstrom::storage::MANAGED);

    std::vector<uint8_t> expected_values = {(uint8_t)17, (uint8_t)22, (uint8_t)3, (uint8_t)20};
    assert_array_equals(static_cast<uint8_t*>(output_values.data()), expected_values.data(), expected_values.size());

    uint64_t i0 = boost::any_cast<uint64_t>(output_indices.get(0));
    assert(i0  == 0 || i0 == 4 || i0 == 6 );

    uint64_t i1 = boost::any_cast<uint64_t>(output_indices.get(1));
    assert(i1  == 1 || i1 == 7 );

    uint64_t i2 = boost::any_cast<uint64_t>(output_indices.get(2));
    assert(i2 == 2 );

    uint64_t i3 = boost::any_cast<uint64_t>(output_indices.get(3));
    assert(i3 == 3 || i3 == 5 );

    // Testing MIN
    std::tie(output_values, output_indices) = maelstrom::reduce_by_key(
        m_keys,
        m_array,
        maelstrom::MIN
    );

    assert( output_values.get_dtype() == maelstrom::uint8 );
    assert( output_indices.get_dtype() == maelstrom::uint64 );

    assert( output_values.size() == 4 );
    assert( output_indices.size() == 4 );

    output_values = output_values.to(maelstrom::storage::MANAGED);
    output_indices = output_indices.to(maelstrom::storage::MANAGED);

    expected_values = {(uint8_t)2, (uint8_t)5, (uint8_t)3, (uint8_t)4};
    assert_array_equals(static_cast<uint8_t*>(output_values.data()), expected_values.data(), expected_values.size());

    std::vector<size_t> expected_indices = {(size_t)0, (size_t)1, (size_t)2, (size_t)5};
    assert_array_equals(static_cast<size_t*>(output_indices.data()), expected_indices.data(), expected_indices.size());

    // Testing MAX
    std::tie(output_values, output_indices) = maelstrom::reduce_by_key(
        m_keys,
        m_array,
        maelstrom::MAX
    );

    assert( output_values.get_dtype() == maelstrom::uint8 );
    assert( output_indices.get_dtype() == maelstrom::uint64 );

    assert( output_values.size() == 4 );
    assert( output_indices.size() == 4 );

    output_values = output_values.to(maelstrom::storage::MANAGED);
    output_indices = output_indices.to(maelstrom::storage::MANAGED);

    expected_values = {(uint8_t)11, (uint8_t)17, (uint8_t)3, (uint8_t)16};
    assert_array_equals(static_cast<uint8_t*>(output_values.data()), expected_values.data(), expected_values.size());

    expected_indices = {(size_t)6, (size_t)7, (size_t)2, (size_t)3};
    assert_array_equals(static_cast<size_t*>(output_indices.data()), expected_indices.data(), expected_indices.size());
}

void test_reduce_mean() {
    std::vector<uint8_t> cpp_array = {(uint8_t)2, (uint8_t)5, (uint8_t)3, (uint8_t)16, (uint8_t)4, (uint8_t)4, (uint8_t)11, (uint8_t)17};
    std::vector<int64_t> cpp_keys =  {(int64_t)2, (int64_t)5, (int64_t)9, (int64_t)13, (int64_t)2, (int64_t)13, (int64_t)2, (int64_t)5};

    maelstrom::vector m_array(
        maelstrom::storage::DEVICE,
        maelstrom::uint8,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    maelstrom::vector m_keys(
        maelstrom::storage::DEVICE,
        maelstrom::int64,
        cpp_keys.data(),
        cpp_keys.size()
    );

    maelstrom::vector output_values;
    maelstrom::vector output_indices;

    // Testing SUM
    std::tie(output_values, output_indices) = maelstrom::reduce_by_key(
        m_keys,
        m_array,
        maelstrom::MEAN
    );

    assert( output_values.get_mem_type() == maelstrom::storage::DEVICE );
    assert( output_indices.get_mem_type() == maelstrom::storage::DEVICE );

    assert( output_values.get_dtype() == maelstrom::float64 );
    assert( output_indices.get_dtype() == maelstrom::uint64 );

    assert( output_values.size() == 4 );
    assert( output_indices.size() == 4 );

    output_values = output_values.to(maelstrom::storage::MANAGED);
    output_indices = output_indices.to(maelstrom::storage::MANAGED);

    std::vector<double> expected_values = {17.0/3.0, 22.0/2.0, 3.0/1.0, 20.0/2.0};
    assert_array_equals(static_cast<double*>(output_values.data()), expected_values.data(), expected_values.size());

    uint64_t i0 = boost::any_cast<uint64_t>(output_indices.get(0));
    assert(i0  == 0 || i0 == 4 || i0 == 6 );

    uint64_t i1 = boost::any_cast<uint64_t>(output_indices.get(1));
    assert(i1  == 1 || i1 == 7 );

    uint64_t i2 = boost::any_cast<uint64_t>(output_indices.get(2));
    assert(i2 == 2 );

    uint64_t i3 = boost::any_cast<uint64_t>(output_indices.get(3));
    assert(i3 == 3 || i3 == 5 );

}