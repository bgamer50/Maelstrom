#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/sparse/query_adjacency.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_query_adjacency_basic_device();
void test_query_adjacency_basic_host();
void test_query_adjacency_reltypes_device();
void test_query_adjacency_reltypes_host();

int main(int argc, char* argv[]) {
    try {
        test_query_adjacency_basic_device();
        test_query_adjacency_basic_host();
        test_query_adjacency_reltypes_device();
        test_query_adjacency_reltypes_host();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_query_adjacency_basic_device() {
    /*
    0  1  0  1  0  0
    1  0  1  0  1  1
    1  0  0  0  0  0
    0  1  1  0  0  0
    0  1  1  1  1  0
    1  0  1  0  0  1
    */
    std::vector<size_t> row = {(size_t)0, (size_t)2, (size_t)6, (size_t)7, (size_t)9, (size_t)13, (size_t)16};
    std::vector<size_t> col = {(size_t)1, (size_t)3, (size_t)0, (size_t)2, (size_t)4, (size_t)5, (size_t)0, (size_t)1, (size_t)2, (size_t)1, (size_t)2, (size_t)3, (size_t)4, (size_t)0, (size_t)2, (size_t)5};

    maelstrom::vector m_row(
        maelstrom::DEVICE,
        maelstrom::uint64,
        row.data(),
        row.size(),
        false
    );

    maelstrom::vector m_col(
        maelstrom::DEVICE,
        maelstrom::uint64,
        col.data(),
        col.size(),
        false
    );

    std::vector<size_t> ix = {(size_t)5, (size_t)0, (size_t)3, (size_t)1};
    maelstrom::vector m_ix(
        maelstrom::DEVICE,
        maelstrom::uint64,
        ix.data(),
        ix.size(),
        false
    );

    maelstrom::vector empty_vec;
    maelstrom::vector origin;
    maelstrom::vector inner;
    std::tie(origin, inner, std::ignore, std::ignore) = maelstrom::sparse::query_adjacency(
        m_row,
        m_col,
        empty_vec,
        empty_vec,
        m_ix,
        empty_vec
    );

    std::vector<size_t> expected_origin = {(size_t)0, (size_t)0, (size_t)0, (size_t)1, (size_t)1, (size_t)2, (size_t)2, (size_t)3, (size_t)3, (size_t)3, (size_t)3};
    std::vector<size_t> expected_inner  = {(size_t)0, (size_t)2, (size_t)5, (size_t)1, (size_t)3, (size_t)1, (size_t)2, (size_t)0, (size_t)2, (size_t)4, (size_t)5};

    assert( origin.get_mem_type() == m_ix.get_mem_type() );
    assert( origin.get_dtype() == maelstrom::uint64 );

    assert( inner.get_mem_type() == m_ix.get_mem_type() );
    assert( inner.get_dtype() == m_ix.get_dtype() );

    origin = origin.to(maelstrom::HOST);
    inner = inner.to(maelstrom::HOST);

    assert_array_equals(static_cast<size_t*>(origin.data()), static_cast<size_t*>(expected_origin.data()), expected_origin.size());
    assert_array_equals(static_cast<size_t*>(inner.data()), static_cast<size_t*>(expected_inner.data()), expected_inner.size());

    std::vector<double> val = {0.6, 0.4, 0.1, 0.3, 0.2, 0.7, 9.1, 3.3, 0.11, 0.44, 0.8, 0.19, 0.66, 0.01, 2.1, 4.1};
    maelstrom::vector m_val(
        maelstrom::DEVICE,
        maelstrom::float64,
        val.data(),
        val.size(),
        false
    );

    std::vector<uint8_t> rel = {(uint8_t)0, (uint8_t)0, (uint8_t)1, (uint8_t)3, (uint8_t)2, (uint8_t)0, (uint8_t)9, (uint8_t)3, (uint8_t)0, (uint8_t)4, (uint8_t)8, (uint8_t)9, (uint8_t)6, (uint8_t)7, (uint8_t)5, (uint8_t)4};
    maelstrom::vector m_rel(
        maelstrom::DEVICE,
        maelstrom::uint8,
        rel.data(),
        rel.size(),
        false
    );

    maelstrom::vector r_val;
    maelstrom::vector r_rel;
    std::tie(origin, inner, r_val, r_rel) = maelstrom::sparse::query_adjacency(
        m_row,
        m_col,
        m_val,
        m_rel,
        m_ix,
        empty_vec,
        false,
        true,
        true
    );

    assert( inner.empty() );

    std::vector<double> expected_val = {0.01, 2.1, 4.1, 0.6, 0.4, 3.3, 0.11, 0.1, 0.3, 0.2, 0.7};
    std::vector<uint8_t> expected_rel = {(uint8_t)7, (uint8_t)5, (uint8_t)4, (uint8_t)0, (uint8_t)0, (uint8_t)3, (uint8_t)0, (uint8_t)1, (uint8_t)3, (uint8_t)2, (uint8_t)0};

    assert( r_val.get_mem_type() == m_val.get_mem_type() );
    assert( r_val.get_dtype() == m_val.get_dtype() );
    assert( r_val.size() == expected_val.size() );

    assert( r_rel.get_mem_type() == m_rel.get_mem_type() );
    assert( r_rel.get_dtype() == m_rel.get_dtype() );
    assert( r_rel.size() == expected_rel.size() );

    r_val = r_val.to(maelstrom::HOST);
    r_rel = r_rel.to(maelstrom::HOST);
    
    assert_array_equals(static_cast<double*>(r_val.data()), expected_val.data(), expected_val.size());
    assert_array_equals(static_cast<uint8_t*>(r_rel.data()), expected_rel.data(), expected_rel.size());
}

void test_query_adjacency_basic_host() {
    /*
    0  1  0  1  0  0
    1  0  1  0  1  1
    1  0  0  0  0  0
    0  1  1  0  0  0
    0  1  1  1  1  0
    1  0  1  0  0  1
    */
    std::vector<int> row = {0, 2, 6, 7, 9, 13, 16};
    std::vector<int> col = {1, 3, 0, 2, 4, 5, 0, 1, 2, 1, 2, 3, 4, 0, 2, 5};

    maelstrom::vector m_row(
        maelstrom::HOST,
        maelstrom::int32,
        row.data(),
        row.size(),
        false
    );

    maelstrom::vector m_col(
        maelstrom::HOST,
        maelstrom::int32,
        col.data(),
        col.size(),
        false
    );

    std::vector<int> ix = {5, 0, 3, 1};
    maelstrom::vector m_ix(
        maelstrom::HOST,
        maelstrom::int32,
        ix.data(),
        ix.size(),
        false
    );

    maelstrom::vector empty_vec;
    maelstrom::vector origin;
    maelstrom::vector inner;
    std::tie(origin, inner, std::ignore, std::ignore) = maelstrom::sparse::query_adjacency(
        m_row,
        m_col,
        empty_vec,
        empty_vec,
        m_ix,
        empty_vec
    );

    std::vector<size_t> expected_origin = {(size_t)0, (size_t)0, (size_t)0, (size_t)1, (size_t)1, (size_t)2, (size_t)2, (size_t)3, (size_t)3, (size_t)3, (size_t)3};
    std::vector<int> expected_inner  = {0, 2, 5, 1, 3, 1, 2, 0, 2, 4, 5};

    assert( origin.get_mem_type() == m_ix.get_mem_type() );
    assert( origin.get_dtype() == maelstrom::uint64 );

    assert( inner.get_mem_type() == m_ix.get_mem_type() );
    assert( inner.get_dtype() == m_ix.get_dtype() );

    assert_array_equals(static_cast<size_t*>(origin.data()), static_cast<size_t*>(expected_origin.data()), expected_origin.size());
    assert_array_equals(static_cast<int*>(inner.data()), static_cast<int*>(expected_inner.data()), expected_inner.size());

    std::vector<double> val = {0.6, 0.4, 0.1, 0.3, 0.2, 0.7, 9.1, 3.3, 0.11, 0.44, 0.8, 0.19, 0.66, 0.01, 2.1, 4.1};
    maelstrom::vector m_val(
        maelstrom::HOST,
        maelstrom::float64,
        val.data(),
        val.size(),
        false
    );

    std::vector<uint8_t> rel = {(uint8_t)0, (uint8_t)0, (uint8_t)1, (uint8_t)3, (uint8_t)2, (uint8_t)0, (uint8_t)9, (uint8_t)3, (uint8_t)0, (uint8_t)4, (uint8_t)8, (uint8_t)9, (uint8_t)6, (uint8_t)7, (uint8_t)5, (uint8_t)4};
    maelstrom::vector m_rel(
        maelstrom::HOST,
        maelstrom::uint8,
        rel.data(),
        rel.size(),
        false
    );

    maelstrom::vector r_val;
    maelstrom::vector r_rel;
    std::tie(origin, inner, r_val, r_rel) = maelstrom::sparse::query_adjacency(
        m_row,
        m_col,
        m_val,
        m_rel,
        m_ix,
        empty_vec,
        false,
        true,
        true
    );

    assert( inner.empty() );

    std::vector<double> expected_val = {0.01, 2.1, 4.1, 0.6, 0.4, 3.3, 0.11, 0.1, 0.3, 0.2, 0.7};
    std::vector<uint8_t> expected_rel = {(uint8_t)7, (uint8_t)5, (uint8_t)4, (uint8_t)0, (uint8_t)0, (uint8_t)3, (uint8_t)0, (uint8_t)1, (uint8_t)3, (uint8_t)2, (uint8_t)0};

    assert( r_val.get_mem_type() == m_val.get_mem_type() );
    assert( r_val.get_dtype() == m_val.get_dtype() );
    assert( r_val.size() == expected_val.size() );

    assert( r_rel.get_mem_type() == m_rel.get_mem_type() );
    assert( r_rel.get_dtype() == m_rel.get_dtype() );
    assert( r_rel.size() == expected_rel.size() );
    
    assert_array_equals(static_cast<double*>(r_val.data()), expected_val.data(), expected_val.size());
    assert_array_equals(static_cast<uint8_t*>(r_rel.data()), expected_rel.data(), expected_rel.size());
}

void test_query_adjacency_reltypes_device() {
    /*
    0  1  0  1  0  0
    1  0  1  0  1  1
    1  0  0  0  0  0
    0  1  1  0  0  0
    0  1  1  1  1  0
    1  0  1  0  0  1
    */
    std::vector<int> row = {0, 2, 6, 7, 9, 13, 16};
    std::vector<int> col = {1, 3, 0, 2, 4, 5, 0, 1, 2, 1, 2, 3, 4, 0, 2, 5};
    std::vector<uint8_t> rel = {(uint8_t)1, (uint8_t)2, (uint8_t)0, (uint8_t)2, (uint8_t)4, (uint8_t)2, (uint8_t)0, (uint8_t)1, (uint8_t)2, (uint8_t)1, (uint8_t)2, (uint8_t)3, (uint8_t)4, (uint8_t)0, (uint8_t)4, (uint8_t)4};

    maelstrom::vector m_row(
        maelstrom::MANAGED,
        maelstrom::int32,
        row.data(),
        row.size(),
        false
    );

    maelstrom::vector m_col(
        maelstrom::MANAGED,
        maelstrom::int32,
        col.data(),
        col.size(),
        false
    );

    maelstrom::vector m_rel(
        maelstrom::MANAGED,
        maelstrom::uint8,
        rel.data(),
        rel.size(),
        false
    );

    std::vector<int> ix = {1, 1, 3, 3, 4, 5};
    maelstrom::vector m_ix(
        maelstrom::MANAGED,
        maelstrom::int32,
        ix.data(),
        ix.size(),
        false
    );

    std::vector<uint8_t> reltypes = {(uint8_t)0, (uint8_t)2};
    maelstrom::vector m_reltypes(
        maelstrom::MANAGED,
        maelstrom::uint8,
        reltypes.data(),
        reltypes.size(),
        false
    );
    
    maelstrom::vector empty_vec;
    maelstrom::vector origin;
    maelstrom::vector inner;
    maelstrom::vector r_rel;
    std::tie(origin, inner, std::ignore, r_rel) = maelstrom::sparse::query_adjacency(
        m_row,
        m_col,
        empty_vec,
        m_rel,
        m_ix,
        m_reltypes,
        true,
        false,
        true
    );

    std::vector<size_t> expected_origin = {(size_t)0, (size_t)0, (size_t)0, (size_t)1, (size_t)1, (size_t)1, (size_t)2, (size_t)3, (size_t)4, (size_t)5};
    std::vector<uint8_t> expected_rel = {(uint8_t)0, (uint8_t)2, (uint8_t)2, (uint8_t)0, (uint8_t)2, (uint8_t)2, (uint8_t)2, (uint8_t)2, (uint8_t)2, (uint8_t)0};
    std::vector<int32_t> expected_inner = {(int32_t)0, (int32_t)2, (int32_t)5, (int32_t)0, (int32_t)2, (int32_t)5, (int32_t)2, (int32_t)2, (int32_t)2, (int32_t)0};

    assert_array_equals(static_cast<size_t*>(origin.data()), expected_origin.data(), expected_origin.size());
    assert_array_equals(static_cast<uint8_t*>(r_rel.data()), expected_rel.data(), expected_rel.size());
    assert_array_equals(static_cast<int32_t*>(inner.data()), expected_inner.data(), expected_inner.size());
}

void test_query_adjacency_reltypes_host() {
    /*
    0  1  0  1  0  0
    1  0  1  0  1  1
    1  0  0  0  0  0
    0  1  1  0  0  0
    0  1  1  1  1  0
    1  0  1  0  0  1
    */
    std::vector<int> row = {0, 2, 6, 7, 9, 13, 16};
    std::vector<int> col = {1, 3, 0, 2, 4, 5, 0, 1, 2, 1, 2, 3, 4, 0, 2, 5};
    std::vector<uint8_t> rel = {(uint8_t)1, (uint8_t)2, (uint8_t)0, (uint8_t)2, (uint8_t)4, (uint8_t)2, (uint8_t)0, (uint8_t)1, (uint8_t)2, (uint8_t)1, (uint8_t)2, (uint8_t)3, (uint8_t)4, (uint8_t)0, (uint8_t)4, (uint8_t)4};

    maelstrom::vector m_row(
        maelstrom::HOST,
        maelstrom::int32,
        row.data(),
        row.size(),
        false
    );

    maelstrom::vector m_col(
        maelstrom::HOST,
        maelstrom::int32,
        col.data(),
        col.size(),
        false
    );

    maelstrom::vector m_rel(
        maelstrom::HOST,
        maelstrom::uint8,
        rel.data(),
        rel.size(),
        false
    );

    std::vector<int> ix = {1, 1, 3, 3, 4, 5};
    maelstrom::vector m_ix(
        maelstrom::HOST,
        maelstrom::int32,
        ix.data(),
        ix.size(),
        false
    );

    std::vector<uint8_t> reltypes = {(uint8_t)0, (uint8_t)2};
    maelstrom::vector m_reltypes(
        maelstrom::HOST,
        maelstrom::uint8,
        reltypes.data(),
        reltypes.size(),
        false
    );
    
    maelstrom::vector empty_vec;
    maelstrom::vector origin;
    maelstrom::vector inner;
    maelstrom::vector r_rel;
    std::tie(origin, inner, std::ignore, r_rel) = maelstrom::sparse::query_adjacency(
        m_row,
        m_col,
        empty_vec,
        m_rel,
        m_ix,
        m_reltypes,
        true,
        false,
        true
    );

    std::vector<size_t> expected_origin = {(size_t)0, (size_t)0, (size_t)0, (size_t)1, (size_t)1, (size_t)1, (size_t)2, (size_t)3, (size_t)4, (size_t)5};
    std::vector<uint8_t> expected_rel = {(uint8_t)0, (uint8_t)2, (uint8_t)2, (uint8_t)0, (uint8_t)2, (uint8_t)2, (uint8_t)2, (uint8_t)2, (uint8_t)2, (uint8_t)0};
    std::vector<int32_t> expected_inner = {(int32_t)0, (int32_t)2, (int32_t)5, (int32_t)0, (int32_t)2, (int32_t)5, (int32_t)2, (int32_t)2, (int32_t)2, (int32_t)0};

    assert_array_equals(static_cast<size_t*>(origin.data()), expected_origin.data(), expected_origin.size());
    assert_array_equals(static_cast<uint8_t*>(r_rel.data()), expected_rel.data(), expected_rel.size());
    assert_array_equals(static_cast<int32_t*>(inner.data()), expected_inner.data(), expected_inner.size());
}