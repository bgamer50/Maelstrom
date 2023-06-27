#pragma once

#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    template <typename T>
    struct filter_fn {
        T filter_val;
        maelstrom::comparator cmp;

        __host__ __device__ bool operator() (const T val) {
            switch(cmp) {
                case EQUALS:
                    return val == filter_val;
                case LESS_THAN:
                    return val < filter_val;
                case GREATER_THAN:
                    return val > filter_val;
                case LESS_THAN_OR_EQUAL:
                    return val <= filter_val;
                case GREATER_THAN_OR_EQUAL:
                    return val >= filter_val;
                case NOT_EQUALS:
                    return val != filter_val;
            }

            return false;
        }
    };

    template <typename T>
    struct compare_fn {
        maelstrom::comparator cmp;
        bool invert;

        __host__ __device__ bool operator() (const T left, const T right) {
            bool z;
            switch(cmp) {
                case EQUALS:
                    z = (left == right);
                    break;
                case LESS_THAN:
                    z = (left < right);
                    break;
                case GREATER_THAN:
                    z = (left > right);
                    break;
                case LESS_THAN_OR_EQUAL:
                    z = (left <= right);
                    break;
                case GREATER_THAN_OR_EQUAL:
                    z = (left >= right);
                    break;
                case NOT_EQUALS:
                    z = (left != right);
                    break;
            }

            return invert ? (!z) : (z);
        }
    };

}