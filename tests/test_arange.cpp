#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/arange.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_arange_basic();

int main(int argc, char* argv[]) {
    try {
        test_arange_basic();
    } catch(std::exception& err) {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_arange_basic() {
    auto v = maelstrom::arange(maelstrom::storage::HOST, 10);
    assert( v.size() == 10 );
    assert( v.get_mem_type() == maelstrom::HOST );
    assert( v.get_dtype() == maelstrom::int32 );
    
    std::vector<int> w = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    assert_array_equals(static_cast<int*>(v.data()), w.data(), 10);

    v = maelstrom::arange(maelstrom::storage::DEVICE, static_cast<int8_t>(-3), static_cast<int8_t>(17));
    assert( v.size() == 20 );
    assert( v.get_mem_type() == maelstrom::DEVICE );
    assert( v.get_dtype() == maelstrom::int8 );
    v = v.to(maelstrom::HOST);
    
    std::vector<int8_t> u = {(int8_t)-3, (int8_t)-2, (int8_t)-1, (int8_t)0, (int8_t)1, (int8_t)2, (int8_t)3, (int8_t)4, (int8_t)5, (int8_t)6, (int8_t)7, (int8_t)8, (int8_t)9, (int8_t)10,(int8_t) (int8_t)11, (int8_t)12, (int8_t)13, (int8_t)14, (int8_t)15, (int8_t)16};
    assert_array_equals(static_cast<int8_t*>(v.data()), u.data(), 20);

    v = maelstrom::arange(maelstrom::storage::MANAGED, static_cast<uint64_t>(1), static_cast<uint64_t>(62), static_cast<uint64_t>(13));
    assert( v.size() ==  5);
    assert( v.get_mem_type() == maelstrom::MANAGED );
    assert( v.get_dtype() == maelstrom::uint64 );

    std::vector<uint64_t> x = {(uint64_t)1, (uint64_t)14, (uint64_t)27, (uint64_t)40, (uint64_t)53};
    assert_array_equals(static_cast<uint64_t*>(v.data()), x.data(), 5);
}