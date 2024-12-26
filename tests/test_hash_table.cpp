#include "maelstrom/containers/vector.h"
#include "maelstrom/containers/hash_table.h"
#include "maelstrom/algorithms/sort.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>
#include <limits>

using namespace maelstrom::test;

void test_hash_table_device();
void test_hash_table_host();

int main(int argc, char* argv[]) {
    try {
        test_hash_table_host();
        test_hash_table_device();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_hash_table_device() {
    std::vector<int> cpp_keys =   {0,       1,    2,    3,    4,    5,    6,    7,    3,    0};
    std::vector<float> cpp_vals = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.1f, 1.3f, 1.5f, 1.7f, 1.9f};

    maelstrom::vector m_keys(
        maelstrom::DEVICE,
        maelstrom::int32,
        cpp_keys.data(),
        cpp_keys.size(),
        false
    );

    maelstrom::vector m_vals(
        maelstrom::DEVICE,
        maelstrom::float32,
        cpp_vals.data(),
        cpp_vals.size(),
        false
    );

    maelstrom::hash_table table(
        maelstrom::DEVICE,
        maelstrom::int32,
        maelstrom::float32
    );

    table.set(
        m_keys,
        m_vals
    );

    assert ( table.size() == 8 );

    float not_found = std::any_cast<float>(table.val_not_found());
    std::vector<int> cpp_ret_keys = {6, 9, 11, -1, 2};
    std::vector<float> cpp_expected_vals = {1.3f, not_found, not_found, not_found, 0.5f};

    maelstrom::vector m_ret_keys(
        maelstrom::DEVICE,
        maelstrom::int32,
        cpp_ret_keys.data(),
        cpp_ret_keys.size(),
        false
    );

    auto m_ret_vals = table.get(m_ret_keys);
    assert( m_ret_vals.get_mem_type() == maelstrom::DEVICE );
    assert( m_ret_vals.get_dtype() == maelstrom::float32 );
    assert( m_ret_vals.size() == 5);

    m_ret_vals = m_ret_vals.to(maelstrom::MANAGED);
    assert_array_equals(static_cast<float*>(m_ret_vals.data()), cpp_expected_vals.data(), cpp_expected_vals.size());

    std::vector<int> exp_keys = {0, 1, 2, 3, 4, 5, 6, 7};
    maelstrom::vector all_keys = table.get_keys().to(maelstrom::HOST);
    maelstrom::sort(all_keys);
    assert_array_equals(exp_keys.data(), static_cast<int*>(all_keys.data()), all_keys.size());

    maelstrom::vector all_vals = table.get_values().to(maelstrom::HOST);
    for(size_t k = 0; k < all_vals.size(); ++k) assert ( std::find(cpp_vals.begin(), cpp_vals.end(), std::any_cast<float>(all_vals.get(k))) != cpp_vals.end() );

    std::vector<int> cpp_cnt_keys = {0, 3, 6, 9};
    std::vector<uint8_t> cpp_cnt_expected = {true, true, true, false};
    
    maelstrom::vector m_cnt_keys(
        maelstrom::MANAGED,
        maelstrom::int32,
        cpp_cnt_keys.data(),
        cpp_cnt_keys.size(),
        false
    );

    maelstrom::vector m_cnt = table.contains(m_cnt_keys);
    assert( m_cnt.get_dtype() == maelstrom::uint8 );
    assert( m_cnt.get_mem_type() == maelstrom::MANAGED );
    assert_array_equals(static_cast<uint8_t*>(m_cnt.data()), cpp_cnt_expected.data(), 4);

    table.remove(m_cnt_keys);
    m_cnt = table.contains(m_cnt_keys);
    for(size_t k = 0; k < m_cnt_keys.size(); ++k) {
        assert( static_cast<uint8_t*>(m_cnt.data())[k] == false );
    }
}

void test_hash_table_host() {
    std::vector<int> cpp_keys =   {0,       1,    2,    3,    4,    5,    6,    7,    3,    0};
    std::vector<float> cpp_vals = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.1f, 1.3f, 1.5f, 1.7f, 1.9f};

    maelstrom::vector m_keys(
        maelstrom::HOST,
        maelstrom::int32,
        cpp_keys.data(),
        cpp_keys.size(),
        false
    );

    maelstrom::vector m_vals(
        maelstrom::HOST,
        maelstrom::float32,
        cpp_vals.data(),
        cpp_vals.size(),
        false
    );

    maelstrom::hash_table table(
        maelstrom::HOST,
        maelstrom::int32,
        maelstrom::float32
    );

    table.set(
        m_keys,
        m_vals
    );

    assert ( table.size() == 8 );

    float not_found = std::any_cast<float>(table.val_not_found());
    std::vector<int> cpp_ret_keys = {6, 9, 11, -1, 2};
    std::vector<float> cpp_expected_vals = {1.3f, not_found, not_found, not_found, 0.5f};

    maelstrom::vector m_ret_keys(
        maelstrom::HOST,
        maelstrom::int32,
        cpp_ret_keys.data(),
        cpp_ret_keys.size(),
        false
    );

    auto m_ret_vals = table.get(m_ret_keys);
    assert( m_ret_vals.get_mem_type() == maelstrom::HOST );
    assert( m_ret_vals.get_dtype() == maelstrom::float32 );
    assert( m_ret_vals.size() == 5);

    assert_array_equals(static_cast<float*>(m_ret_vals.data()), cpp_expected_vals.data(), cpp_expected_vals.size());

    std::vector<int> exp_keys = {0, 1, 2, 3, 4, 5, 6, 7};
    maelstrom::vector all_keys = table.get_keys();
    maelstrom::sort(all_keys);
    assert_array_equals(exp_keys.data(), static_cast<int*>(all_keys.data()), all_keys.size());

    maelstrom::vector all_vals = table.get_values().to(maelstrom::HOST);
    for(size_t k = 0; k < all_vals.size(); ++k) assert ( std::find(cpp_vals.begin(), cpp_vals.end(), std::any_cast<float>(all_vals.get(k))) != cpp_vals.end() );
    
    std::vector<int> cpp_cnt_keys = {0, 3, 6, 9};
    std::vector<uint8_t> cpp_cnt_expected = {true, true, true, false};
    
    maelstrom::vector m_cnt_keys(
        maelstrom::HOST,
        maelstrom::int32,
        cpp_cnt_keys.data(),
        cpp_cnt_keys.size(),
        false
    );

    maelstrom::vector m_cnt = table.contains(m_cnt_keys);
    assert( m_cnt.get_dtype() == maelstrom::uint8 );
    assert( m_cnt.get_mem_type() == maelstrom::HOST );
    assert_array_equals(static_cast<uint8_t*>(m_cnt.data()), cpp_cnt_expected.data(), 4);

    table.remove(m_cnt_keys);
    m_cnt = table.contains(m_cnt_keys);
    for(size_t k = 0; k < m_cnt_keys.size(); ++k) {
        assert( static_cast<uint8_t*>(m_cnt.data())[k] == false );
    }
}
