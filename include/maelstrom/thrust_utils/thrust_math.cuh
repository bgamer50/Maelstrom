#pragma once

#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {
    template <typename T>
    struct math_binary_plus : public thrust::unary_function<thrust::tuple<T, T>, T> {
        __device__ __host__ T operator()(const thrust::tuple<T, T> input) {
            return thrust::get<0>(input) + thrust::get<1>(input);
        }
    };

    template <typename T>
    struct math_binary_minus : public thrust::unary_function<thrust::tuple<T, T>, T> {
        __device__ __host__ T operator()(const thrust::tuple<T, T> input) {
            return thrust::get<0>(input) - thrust::get<1>(input);
        }
    };

    template <typename T>
    struct math_binary_times : public thrust::unary_function<thrust::tuple<T, T>, T> {
        __device__ __host__ T operator()(const thrust::tuple<T, T> input) {
            return thrust::get<0>(input) * thrust::get<1>(input);
        }
    };

    template <typename T>
    struct math_binary_divide : public thrust::unary_function<thrust::tuple<T, T>, T> {
        __device__ __host__ T operator()(const thrust::tuple<T, T> input) {
            return thrust::get<0>(input) / thrust::get<1>(input);
        }
    };
};