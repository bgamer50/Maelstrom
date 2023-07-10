#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/sparse/search_sorted_sparse.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_search_sorted_sparse_device();
void test_search_sorted_sparse_host();

int main(int argc, char* argv[]) {
    try {
        test_search_sorted_sparse_device();
        test_search_sorted_sparse_host();
    } catch(std::exception& err) {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_search_sorted_sparse_device() {
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

    std::vector<size_t> ix_r = {(size_t)0, (size_t)2, (size_t)4, (size_t)5, (size_t)1, (size_t)2, (size_t)1};
    std::vector<size_t> ix_c = {(size_t)0, (size_t)2, (size_t)1, (size_t)5, (size_t)3, (size_t)0, (size_t)5};

    maelstrom::vector m_ix_r(
        maelstrom::DEVICE,
        maelstrom::uint64,
        ix_r.data(),
        ix_r.size(),
        false
    );

    maelstrom::vector m_ix_c(
        maelstrom::DEVICE,
        maelstrom::uint64,
        ix_c.data(),
        ix_c.size(),
        false
    );

    auto result = maelstrom::sparse::search_sorted_sparse(
        m_row,
        m_col,
        m_ix_r,
        m_ix_c,
        99
    );

    std::vector<size_t> expected = {(size_t)99, (size_t)99, (size_t)9, (size_t)15, (size_t)99, (size_t)6, (size_t)5};

    assert( result.size() == expected.size() );
    assert( result.get_dtype() == maelstrom::uint64 );
    assert( result.get_mem_type() == maelstrom::DEVICE );
    
    result = result.to(maelstrom::HOST);
    assert_array_equals(static_cast<size_t*>(result.data()), expected.data(), expected.size());

}

void test_search_sorted_sparse_host() {
 /*
    0  1  0  1  0  0
    1  0  1  0  1  1
    1  0  0  0  0  0
    0  1  1  0  0  0
    0  1  1  1  1  0
    1  0  1  0  0  1
    */
    std::vector<int32_t> row = {0, 2, 6, 7, 9, 13, 16};
    std::vector<int32_t> col = {1, 3, 0, 2, 4, 5, 0, 1, 2, 1, 2, 3, 4, 0, 2, 5};

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

    std::vector<int32_t> ix_r = {0, 2, 4, 5, 1, 2, 1};
    std::vector<int32_t> ix_c = {0, 2, 1, 5, 3, 0, 5};

    maelstrom::vector m_ix_r(
        maelstrom::HOST,
        maelstrom::int32,
        ix_r.data(),
        ix_r.size(),
        false
    );

    maelstrom::vector m_ix_c(
        maelstrom::HOST,
        maelstrom::int32,
        ix_c.data(),
        ix_c.size(),
        false
    );

    auto result = maelstrom::sparse::search_sorted_sparse(
        m_row,
        m_col,
        m_ix_r,
        m_ix_c,
        -1
    );

    std::vector<int32_t> expected = {-1, -1, 9, 15, -1, 6, 5};

    assert( result.size() == expected.size() );
    assert( result.get_dtype() == maelstrom::int32 );
    assert( result.get_mem_type() == maelstrom::HOST );
    
    assert_array_equals(static_cast<int32_t*>(result.data()), expected.data(), expected.size());
}