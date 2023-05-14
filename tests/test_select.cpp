#include "containers/vector.h"
#include "algorithms/select.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_select_basic(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    try {
        for(maelstrom::storage storage : {maelstrom::storage::DEVICE, maelstrom::storage::HOST, maelstrom::storage::PINNED}) {
            test_select_basic(storage);
        }
    } catch(std::exception& err) {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_select_basic(maelstrom::storage storage) {
    std::vector<uint8_t> cpp_array = {(uint8_t)0, (uint8_t)1, (uint8_t)2, (uint8_t)3, (uint8_t)4, (uint8_t)5, (uint8_t)6, (uint8_t)7};

    maelstrom::vector m_array(
        storage,
        maelstrom::uint8,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    std::vector<int32_t> cpp_select = {1, 1, 2, 2, 6, 7, 0, 3, 3};

    maelstrom::vector m_select(
        storage,
        maelstrom::int32,
        cpp_select.data(),
        cpp_select.size(),
        false
    );

    m_array = maelstrom::select(m_array, m_select);
    assert( m_array.get_mem_type() == storage );
    assert( m_array.get_dtype() == maelstrom::uint8 );
    assert( m_array.size() == 9 );
    m_array = m_array.to(maelstrom::storage::HOST);

    std::vector<uint8_t> expected_results = {(uint8_t)1, (uint8_t)1, (uint8_t)2, (uint8_t)2, (uint8_t)6, (uint8_t)7, (uint8_t)0, (uint8_t)3, (uint8_t)3};
    assert_array_equals(
        static_cast<uint8_t*>(m_array.data()),
        expected_results.data(),
        m_array.size()
    );
}