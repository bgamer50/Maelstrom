#pragma once

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/gather.h>

#include <thrust/adjacent_difference.h>
#include <thrust/set_operations.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>

#include <thrust/unique.h>

#include <thrust/sort.h>

#include <thrust/reduce.h>

#include <thrust/execution_policy.h>

#include <limits>

#include "thrust_utils/execution.cuh"
#include "storage/comparison.h"

namespace maelstrom {

    template <typename T>
    struct unary_plus_op : public thrust::unary_function<T, T> {
        T plus_val;
        
        __device__ __host__ T operator()(T in) const {
            return in + plus_val;
        }
    };

    template <typename T>
    struct unary_times_op : public thrust::unary_function<T, T> {
        T times_val;
        
        __device__ __host__ T operator()(T in) const {
            return in * times_val;
        }
    };

    struct plus_op : public thrust::unary_function<thrust::tuple<size_t,size_t>,size_t> {
        __device__ size_t operator()(thrust::tuple<size_t, size_t> t) const {
            return thrust::get<0>(t) + thrust::get<1>(t);
        }
    };

    struct minus_op : public thrust::unary_function<thrust::tuple<size_t,size_t>,size_t> {
        __device__ size_t operator()(thrust::tuple<size_t, size_t> t) const {
            return thrust::get<0>(t) - thrust::get<1>(t);
        }
    };

    template<typename T>
    struct is_max_val {
        __host__ __device__ bool operator() (const T val) {
            return val == std::numeric_limits<T>::max();
        }
    };

    template<typename T>
    inline thrust::device_ptr<T> device_tptr_cast(void* v) {
        return thrust::device_pointer_cast<T>(
            static_cast<T*>(v)
        );
    }

    template <typename I, typename O>
    struct cast_fn {
        __host__ __device__ O operator() (const I val) {
            return static_cast<O>(val);
        }
    };

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
}