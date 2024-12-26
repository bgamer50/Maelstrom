#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/topk.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_topk_basic(maelstrom::storage storage);
void test_topk_small(maelstrom::storage storage);

int main(int argc, char* argv[]) {
    try {
        for(maelstrom::storage storage : {maelstrom::storage::DEVICE, maelstrom::storage::HOST, maelstrom::storage::PINNED}) {
            test_topk_basic(storage);
            test_topk_small(storage);
        }
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_topk_basic(maelstrom::storage storage) {
    std::vector<uint8_t> cpp_array = {(uint8_t)1, (uint8_t)0, (uint8_t)1, (uint8_t)10, (uint8_t)2, (uint8_t)3, (uint8_t)4, (uint8_t)9, (uint8_t)5, (uint8_t)6, (uint8_t)7};

    maelstrom::vector m_array(
        storage,
        maelstrom::uint8,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    auto top_3 = maelstrom::topk(m_array, 3);
    assert( top_3.get_mem_type() == storage );
    assert( top_3.get_dtype() == maelstrom::uint64 );
    
    std::vector<size_t> exp_top3 = {3, 7, 10};
    top_3 = top_3.to(maelstrom::HOST);  
    assert_array_equals(static_cast<size_t*>(top_3.data()), exp_top3.data(), exp_top3.size());

    auto bot_3 = maelstrom::topk(m_array, 3, true);
    assert( bot_3.get_mem_type() == storage );
    assert( bot_3.get_dtype() == maelstrom::uint64 );
    
    // multiple valid options
    assert( std::any_cast<size_t>(bot_3.get(0)) == 1 );
    assert( 
        (std::any_cast<size_t>(bot_3.get(1)) == 0 && std::any_cast<size_t>(bot_3.get(2)) == 2) ||
        (std::any_cast<size_t>(bot_3.get(1)) == 2 && std::any_cast<size_t>(bot_3.get(2)) == 0)
    );
}

void test_topk_small(maelstrom::storage storage) {
    std::vector<uint8_t> cpp_array = {(uint8_t)1, (uint8_t)0, (uint8_t)7};

    maelstrom::vector m_array(
        storage,
        maelstrom::uint8,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    auto top_3 = maelstrom::topk(m_array, 4);
    assert( top_3.get_mem_type() == storage );
    assert( top_3.get_dtype() == maelstrom::uint64 );
    
    std::vector<size_t> exp_top3 = {2, 0, 1};
    top_3 = top_3.to(maelstrom::HOST);  
    assert_array_equals(static_cast<size_t*>(top_3.data()), exp_top3.data(), exp_top3.size());
}