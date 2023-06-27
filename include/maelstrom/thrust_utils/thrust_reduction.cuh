#pragma once

#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    template <typename T>
    struct red_min : public thrust::binary_function<T, T, T> {
        __device__ __host__ thrust::tuple<T, size_t> operator()(const thrust::tuple<T, size_t>& left, const thrust::tuple<T, size_t>& right) {
            if(thrust::get<1>(left) == std::numeric_limits<size_t>::max()) return right;
            if(thrust::get<1>(right) == std::numeric_limits<size_t>::max()) return left;

            return thrust::make_tuple(
                (thrust::get<0>(left) <= thrust::get<0>(right)) ? thrust::get<0>(left) : thrust::get<0>(right),
                (thrust::get<0>(left) <= thrust::get<0>(right)) ? thrust::get<1>(left) : thrust::get<1>(right)
            );
        }
    };

    template <typename T>
    struct red_max : thrust::binary_function<thrust::tuple<T, size_t>, thrust::tuple<T, size_t>, thrust::tuple<T, size_t>> {
        __device__ __host__ thrust::tuple<T, size_t> operator()(const thrust::tuple<T, size_t>& left, const thrust::tuple<T, size_t>& right) {
            if(thrust::get<1>(left) == std::numeric_limits<size_t>::max()) return right;
            if(thrust::get<1>(right) == std::numeric_limits<size_t>::max()) return left;
            
            return thrust::make_tuple(
                (thrust::get<0>(left) >= thrust::get<0>(right)) ? thrust::get<0>(left) : thrust::get<0>(right),
                (thrust::get<0>(left) >= thrust::get<0>(right)) ? thrust::get<1>(left) : thrust::get<1>(right)
            );
        }
    };

    template <typename T>
    struct red_sum : public thrust::binary_function<thrust::tuple<T, size_t>, thrust::tuple<T, size_t>, thrust::tuple<T, size_t>> {
        __device__ __host__ thrust::tuple<T, size_t> operator()(const thrust::tuple<T, size_t>& left, const thrust::tuple<T, size_t>& right) {
            if(thrust::get<1>(left) == std::numeric_limits<size_t>::max()) return right;
            if(thrust::get<1>(right) == std::numeric_limits<size_t>::max()) return left;

            return thrust::make_tuple(
                thrust::get<0>(left) + thrust::get<0>(right),
                thrust::get<1>(left)
            );
        }
    };

    template <typename T>
    struct red_product : public thrust::binary_function<thrust::tuple<T, size_t>, thrust::tuple<T, size_t>, thrust::tuple<T, size_t>> {
        __device__ __host__ thrust::tuple<T, size_t> operator()(const thrust::tuple<T, size_t>& left, const thrust::tuple<T, size_t>& right) {
            if(thrust::get<1>(left) == std::numeric_limits<size_t>::max()) return right;
            if(thrust::get<1>(right) == std::numeric_limits<size_t>::max()) return left;

            return thrust::make_tuple(
                thrust::get<0>(left) * thrust::get<0>(right),
                thrust::get<1>(left)
            );
        }
    };

}