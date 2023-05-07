#pragma once

#include <assert.h>
#include <vector>
#include <iostream>

namespace maelstrom {
    namespace test {
        
        template <typename T>
        void assert_vector_equals(std::vector<T>& left, std::vector<T>& right) {
            assert( left.size() == right.size() );
            for(size_t k = 0; k < left.size(); ++k) {
                assert( left[k] == right[k] );
            }
        }

        template <typename T>
        void assert_array_equals(T* left, T* right, size_t size) {
            for(size_t i = 0; i < size; ++i) {
                assert( left[i] == right[i] );
            }
        }
    }
}