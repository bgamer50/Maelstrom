#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/sparse/csr_to_coo.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_csr_to_coo_device();
void test_csr_to_coo_host();

int main(int argc, char* argv[]) {
    try {
        test_csr_to_coo_device();
        test_csr_to_coo_host();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_csr_to_coo_device() {
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
        maelstrom::MANAGED,
        maelstrom::int32,
        row.data(),
        row.size(),
        false
    );

    m_row = maelstrom::sparse::csr_to_coo(
        m_row,
        col.size()
    );

    assert( m_row.size() == 16 );
    assert( m_row.get_mem_type() == m_row.get_mem_type() );
    assert( m_row.get_dtype() == m_row.get_dtype() );

    std::vector<int> expected = {0, 0, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5};
    assert_array_equals(static_cast<int*>(m_row.data()), expected.data(), expected.size());
}

void test_csr_to_coo_host() {
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

    m_row = maelstrom::sparse::csr_to_coo(
        m_row,
        col.size()
    );

    assert( m_row.size() == 16 );
    assert( m_row.get_mem_type() == m_row.get_mem_type() );
    assert( m_row.get_dtype() == m_row.get_dtype() );

    std::vector<int> expected = {0, 0, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5};
    assert_array_equals(static_cast<int*>(m_row.data()), expected.data(), expected.size());
}