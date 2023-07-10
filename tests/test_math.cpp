#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/math.h"
#include "maelstrom/algorithms/set.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>
#include <cmath>

using namespace maelstrom::test;

void test_math_binary();

int main(int argc, char* argv[]) {
    try {
        test_math_binary();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_math_binary() {
    std::vector<int> cpp_array = {0, 1, 2, 3, 4, 5, 6, 7};

    maelstrom::vector m_array(
        maelstrom::storage::DEVICE,
        maelstrom::int32,
        cpp_array.data(),
        cpp_array.size(),
        false
    );

    auto sum = m_array + m_array;
    sum = sum.to(maelstrom::HOST);

    std::vector expected_sum = {0, 2, 4, 6, 8, 10, 12, 14};
    assert_array_equals(static_cast<int*>(sum.data()), expected_sum.data(), expected_sum.size());

    std::vector<float> cpp_fl = {0.1f, 0.3f, 0.5f, 0.2f, -0.5f, -16.6f};

    maelstrom::vector m_fl_array(
        maelstrom::storage::MANAGED,
        maelstrom::float32,
        cpp_fl.data(),
        cpp_fl.size(),
        false
    );

    maelstrom::vector m_d(
        maelstrom::storage::MANAGED,
        maelstrom::float32,
        cpp_fl.size()
    );
    maelstrom::set(m_d, static_cast<float>(64.4));

    auto d_r = m_fl_array / m_d;

    std::vector<float> expected_r = {0.00155f, 0.00466f, 0.00776f, 0.00311f, -0.00776f, -0.25776f};
    for(size_t k = 0; k < expected_r.size(); ++k) assert( std::fabs(*(static_cast<float*>(d_r.data())+k) - expected_r[k]) < 0.00001f );
}
