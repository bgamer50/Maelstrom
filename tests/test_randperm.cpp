#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/randperm.h"
#include "maelstrom/algorithms/reduce.h"
#include "test_utils.hpp"

#include <vector>
#include <set>
#include <iostream>

using namespace maelstrom::test;

void test_randperm_basic(maelstrom::storage mem_type);

int main(int argc, char* argv[]) {
    try {
        test_randperm_basic(maelstrom::HOST);
        test_randperm_basic(maelstrom::DEVICE);
        test_randperm_basic(maelstrom::MANAGED);
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_randperm_basic(maelstrom::storage mem_type) {
    auto perm = maelstrom::randperm(mem_type, 32, 32, 62);

    assert( perm.size() == 32 );

    auto min = std::any_cast<size_t>(maelstrom::reduce(perm, maelstrom::reductor::MIN).first);
    auto max = std::any_cast<size_t>(maelstrom::reduce(perm, maelstrom::reductor::MAX).first);

    assert( min >= 0 );
    assert( max < 32 );

    perm = perm.to(maelstrom::HOST);

    std::set<size_t> s;
    for(size_t k = 0; k < perm.size(); ++k) s.insert(std::any_cast<size_t>(perm.get(k)));

    assert( s.size() == perm.size() );

    perm = maelstrom::randperm(mem_type, 32, 19, 62);
    assert( perm.size() == 19 );
}
