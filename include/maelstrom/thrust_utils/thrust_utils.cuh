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
#include <thrust/binary_search.h>

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

#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/storage/comparison.h"

namespace maelstrom {

    template <typename T>
    struct unary_plus_op : public thrust::unary_function<T, T> {
        T plus_val;
        bool subtract=false;
        
        __device__ __host__ T operator()(T in) const {
            return subtract ? (in - plus_val) : (in + plus_val);
        }
    };

    template <typename T>
    struct unary_times_op : public thrust::unary_function<T, T> {
        T times_val;
        
        __device__ __host__ T operator()(T in) const {
            return in * times_val;
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

}