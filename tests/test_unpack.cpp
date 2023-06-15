#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/unpack.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_unpack_basic();

int main(int argc, char* argv[]) {
    try {
        test_unpack_basic();
    } catch(std::exception& err) {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_unpack_basic() {
    std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7};

    maelstrom::vector m_data(
        maelstrom::storage::MANAGED,
        maelstrom::int32,
        data.data(),
        data.size(),
        false
    );

    auto unpacked_m_data = maelstrom::unpack(m_data);

    for(auto& v : unpacked_m_data) assert( v.size() == 1 );

    for(size_t k = 0; k < unpacked_m_data.size(); ++k) {
        std::cout << k << ": " << boost::any_cast<int32_t>(unpacked_m_data[k].get(0)) << std::endl;
        assert( 
            boost::any_cast<int32_t>(unpacked_m_data[k].get(0)) == data[k]
        );
    }
}