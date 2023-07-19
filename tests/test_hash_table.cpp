#include "maelstrom/containers/vector.h"
#include "maelstrom/containers/hash_table.h"
#include "test_utils.hpp"

#include <vector>
#include <iostream>

using namespace maelstrom::test;

void test_hash_table_device();
void test_hash_table_host();

int main(int argc, char* argv[]) {
    try {
        test_hash_table_device();
        test_hash_table_host();
    } catch(std::exception& err) {
        std::cerr << "FAIL!" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE!" << std::endl;
}

void test_hash_table_device() {
    std::vector<int> cpp_keys =   {0,       1,    2,    3,    4,    5,    6,    7,    3,    0};
    std::vector<float> cpp_vals = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.1f, 1.3f, 1.5f, 1.7f, 1.9f};



    maelstrom::hash_table table(
        maelstrom::DEVICE,
        maelstrom::int32,
        maelstrom::float32
    );


}

void test_hash_table_host() {

}
