#include "maelstrom/containers/vector.h"
#include "maelstrom/containers/sparse_matrix.h"
#include "maelstrom/algorithms/filter.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/search_sorted.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_sparse_matrix_basic();
void test_sparse_matrix_conversion();

int main(int argc, char* argv[]) {
    try {
        test_sparse_matrix_basic();
        test_sparse_matrix_conversion();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_sparse_matrix_basic() {
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

    auto matrix = maelstrom::basic_sparse_matrix(
        m_row,
        m_col,
        maelstrom::vector(),
        std::move(m_rel),
        maelstrom::sparse_matrix_format::CSR,
        5,
        5,
        true
    );

    assert( matrix.is_sorted() );
    assert( matrix.get_format() == maelstrom::CSR );
    assert( matrix.has_relations() );
    assert( !matrix.has_values() );
    assert( matrix.num_rows() == 5 );
    assert( matrix.num_cols() == 5 );
    assert( matrix.num_nonzero() == 16 );

    std::vector<int> ix_r_2d = {0, 0, 1, 2, 3, 4, 5, 1, 1};
    maelstrom::vector m_ix_r(
        maelstrom::MANAGED,
        maelstrom::int32,
        ix_r_2d.data(),
        ix_r_2d.size(),
        false
    );

    std::vector<int> ix_c_2d = {0, 1, 0, 0, 2, 5, 0, 4, 5};
    maelstrom::vector m_ix_c(
        maelstrom::MANAGED,
        maelstrom::int32,
        ix_c_2d.data(),
        ix_c_2d.size(),
        false
    );

    auto r_ix_1d = matrix.get_1d_index_from_2d_index(
        m_ix_r,
        m_ix_c,
        -2
    );

    std::vector<int> expected_ix_1d = {-2, 0, 2, 6, 8, -2, 13, 4, 5};
    assert_array_equals(static_cast<int*>(r_ix_1d.data()), expected_ix_1d.data(), expected_ix_1d.size());

    auto filter_ix = maelstrom::filter(r_ix_1d, maelstrom::NOT_EQUALS, -2);
    
    m_ix_r = maelstrom::select(m_ix_r, filter_ix);
    m_ix_c = maelstrom::select(m_ix_c, filter_ix);
    r_ix_1d = maelstrom::select(r_ix_1d, filter_ix);

    maelstrom::vector r_rows;
    maelstrom::vector r_cols;
    maelstrom::vector r_vals;
    maelstrom::vector r_rels;
    std::tie(r_rows, r_cols, r_vals, r_rels) = matrix.get_entries_1d(r_ix_1d);

    assert_array_equals(static_cast<int*>(r_rows.data()), static_cast<int*>(m_ix_r.data()), m_ix_r.size());
    assert_array_equals(static_cast<int*>(r_cols.data()), static_cast<int*>(m_ix_c.data()), m_ix_c.size());

    std::vector<uint8_t> expected_rels = {(uint8_t)1, (uint8_t)0, (uint8_t)0, (uint8_t)2, (uint8_t)0, (uint8_t)4, (uint8_t)2};
    assert_array_equals(static_cast<uint8_t*>(r_rels.data()), expected_rels.data(), expected_rels.size());

    assert( r_vals.empty() );

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
    std::tie(origin, inner, std::ignore, r_rel) = matrix.query_adjacency(
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

void test_sparse_matrix_conversion() {
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

    auto matrix = maelstrom::basic_sparse_matrix(
        m_row,
        m_col,
        maelstrom::vector(),
        m_rel,
        maelstrom::sparse_matrix_format::CSR,
        6,
        6,
        true
    );

    // convert CSR -> COO
    matrix.to_coo();
    std::vector<int> expected_row_coo = {0, 0, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5};
    auto m_r = matrix.get_row();
    assert_array_equals(static_cast<int*>(m_r.data()), expected_row_coo.data(), expected_row_coo.size());

    // convert COO -> CSC
    matrix.to_csc();

    std::vector<int> expected_row_csc = {1, 2, 5, 0, 3, 4, 1, 3, 4, 5, 0, 4, 1, 4, 1, 5};
    m_r = matrix.get_row();
    assert_array_equals(static_cast<int*>(m_r.data()), expected_row_csc.data(), expected_row_csc.size());

    std::vector<int> expected_col_csc = {0, 3, 6, 10, 12, 14, 16};
    auto m_c = matrix.get_col();
    assert_array_equals(static_cast<int*>(m_c.data()), expected_col_csc.data(), expected_col_csc.size());

    // convert CSC -> CSR
    matrix.to_csr();
    m_r = matrix.get_row();
    assert_array_equals(static_cast<int*>(m_r.data()), static_cast<int*>(m_row.data()), m_row.size());
    m_c = matrix.get_col();
    assert_array_equals(static_cast<int*>(m_c.data()), static_cast<int*>(m_col.data()), m_col.size());
    auto m_re = matrix.get_rel();
    assert_array_equals(static_cast<uint8_t*>(m_re.data()), static_cast<uint8_t*>(m_rel.data()), m_rel.size());
}