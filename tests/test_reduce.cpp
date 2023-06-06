#include "containers/vector.h"
#include "algorithms/reduce.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_reduce_basic();
void test_reduce_ix();

int main(int argc, char* argv[]) {
    try {
        test_reduce_basic();
        test_reduce_ix();
    } catch(std::exception& err) {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_reduce_basic() {
    std::vector<int> cpp_array = {0, 1, 2, 3, 4, 5, 6, 7};

    maelstrom::vector m_array(
        maelstrom::storage::DEVICE,
        maelstrom::int32,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    assert( boost::any_cast<int>(maelstrom::reduce(m_array, maelstrom::reductor::MIN).first) == 0);
    assert( boost::any_cast<int>(maelstrom::reduce(m_array, maelstrom::reductor::MAX).first) == 7);
    assert( boost::any_cast<int>(maelstrom::reduce(m_array, maelstrom::reductor::SUM).first) == 28);
    assert( boost::any_cast<int>(maelstrom::reduce(m_array, maelstrom::reductor::PRODUCT).first) == 0);
    assert( boost::any_cast<double>(maelstrom::reduce(m_array, maelstrom::reductor::MEAN).first) - 3.5 <= 0.001);
}

void test_reduce_ix() {
    std::vector<double> cpp_array = {0.1, 0.2, 0.5, 0.88, 0.03, 0.29, -0.7, -3.3, 9.4};

    maelstrom::vector m_array(
        maelstrom::storage::DEVICE,
        maelstrom::float64,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    boost::any val;
    size_t ix;
    std::tie(val, ix) = maelstrom::reduce(m_array, maelstrom::reductor::MIN);

    assert( boost::any_cast<double>(val) == -3.3 );
    assert( ix ==  7);
    
    boost::any val;
    size_t ix;
    std::tie(val, ix) = maelstrom::reduce(m_array, maelstrom::reductor::MAX);

    assert( boost::any_cast<double>(val) == 9.4 );
    assert( ix ==  8);

    // sum, mean, product have no index guarantee
}
