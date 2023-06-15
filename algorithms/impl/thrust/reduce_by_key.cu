#include "maelstrom/algorithms/reduce_by_key.h"
#include "maelstrom/algorithms/sort.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/count_unique.h"

#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    template <typename E, typename K, typename V>
    std::pair<maelstrom::vector, maelstrom::vector> t_reduce_by_key_from_sorted_vec(E exec_policy, maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, size_t max_unique_keys) {
        maelstrom::vector output_values = maelstrom::vector(input_values.get_mem_type(), input_values.get_dtype(), max_unique_keys);
        maelstrom::vector output_indices = maelstrom::vector(input_values.get_mem_type(), uint64, max_unique_keys);

        auto zip_output_values_ix = thrust::make_zip_iterator(
            maelstrom::device_tptr_cast<V>(output_values.data()),
            maelstrom::device_tptr_cast<size_t>(output_indices.data())
        );

        auto zip_input_values_ix = thrust::make_zip_iterator(
            maelstrom::device_tptr_cast<V>(input_values.data()),
            thrust::make_counting_iterator<size_t>(static_cast<size_t>(0))
        );

        size_t new_size;
        switch(red) {
            case MIN: {
                auto new_end = thrust::reduce_by_key(
                    exec_policy,
                    maelstrom::device_tptr_cast<K>(input_keys.data()),
                    maelstrom::device_tptr_cast<K>(input_keys.data()) + input_keys.size(),
                    zip_input_values_ix,
                    thrust::make_discard_iterator(),
                    zip_output_values_ix,
                    thrust::equal_to<K>(),
                    red_min<V>()
                );
                
                new_size = new_end.second - zip_output_values_ix;
                break;
            }
            case MAX: {
                auto new_end = thrust::reduce_by_key(
                    exec_policy,
                    maelstrom::device_tptr_cast<K>(input_keys.data()),
                    maelstrom::device_tptr_cast<K>(input_keys.data()) + input_keys.size(),
                    zip_input_values_ix,
                    thrust::make_discard_iterator(),
                    zip_output_values_ix,
                    thrust::equal_to<K>(),
                    red_max<V>()
                );
                
                new_size = new_end.second - zip_output_values_ix;
                break;
            }
            case SUM: {
                auto new_end = thrust::reduce_by_key(
                    exec_policy,
                    maelstrom::device_tptr_cast<K>(input_keys.data()),
                    maelstrom::device_tptr_cast<K>(input_keys.data()) + input_keys.size(),
                    zip_input_values_ix,
                    thrust::make_discard_iterator(),
                    zip_output_values_ix,
                    thrust::equal_to<K>(),
                    red_sum<V>()
                );
                
                new_size = new_end.second - zip_output_values_ix;
                break;
            }
            case PRODUCT: {
                auto new_end = thrust::reduce_by_key(
                    exec_policy,
                    maelstrom::device_tptr_cast<K>(input_keys.data()),
                    maelstrom::device_tptr_cast<K>(input_keys.data()) + input_keys.size(),
                    zip_input_values_ix,
                    thrust::make_discard_iterator(),
                    zip_output_values_ix,
                    thrust::equal_to<K>(),
                    red_product<V>()
                );
                
                new_size = new_end.second - zip_output_values_ix;
                break;
            }
            default: {
                throw std::runtime_error("Unsupported reduction type");
            }
        }

        output_values.resize(new_size);
        output_indices.resize(new_size);

        return std::make_pair(
            std::move(output_values),
            std::move(output_indices)
        );
    }

    template <typename E, typename K, typename V> 
    std::pair<maelstrom::vector, maelstrom::vector> t_reduce_mean_by_key_from_sorted_vec(E exec_policy, maelstrom::vector& input_keys, maelstrom::vector& input_values, size_t max_unique_keys) {
        maelstrom::vector output_values;
        maelstrom::vector output_indices;
        std::tie(output_values, output_indices) = t_reduce_by_key_from_sorted_vec<E, K, V>(
            exec_policy,
            input_keys,
            input_values,
            maelstrom::reductor::SUM,
            max_unique_keys
        );

        // sorting shouldn't be necessary since reduce_by_key should preserve order
        // this also means we don't need to know what the keys are

        maelstrom::vector unique_counts;
        std::tie(std::ignore, unique_counts) = maelstrom::count_unique(
            input_keys,
            true
        );

        // Make sure we cast to doubles before doing division
        output_values = output_values.astype(float64);
        unique_counts = unique_counts.astype(float64);

        output_values = output_values / unique_counts;

        return std::make_pair(
            std::move(output_values),
            std::move(output_indices)
        );
    }

    template<typename E, typename K, typename V>
    std::pair<maelstrom::vector, maelstrom::vector> reduce_by_key_from_sorted_vec_dispatch_red(E exec_policy, maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, size_t max_unique_keys) {
        switch(red) {
            case MIN:
            case MAX:
            case SUM:
            case PRODUCT:
                return t_reduce_by_key_from_sorted_vec<E, K, V>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case MEAN: {
                return t_reduce_mean_by_key_from_sorted_vec<E, K, V>(
                    exec_policy,
                    input_keys,
                    input_values,
                    max_unique_keys
                );
            }
        }

        throw std::runtime_error("invalid reductor provided to reduce_by_key_from_sorted_vec");
    }

    template <typename E, typename K>
    std::pair<maelstrom::vector, maelstrom::vector> reduce_by_key_from_sorted_vec_dispatch_val(E exec_policy, maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, size_t max_unique_keys) {
        switch(input_values.get_dtype().prim_type) {
            case UINT64:
                return reduce_by_key_from_sorted_vec_dispatch_red<E, K, uint64_t>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case UINT32:
                return reduce_by_key_from_sorted_vec_dispatch_red<E, K, uint32_t>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case UINT8:
                return reduce_by_key_from_sorted_vec_dispatch_red<E, K, uint8_t>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case INT64:
                return reduce_by_key_from_sorted_vec_dispatch_red<E, K, int64_t>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case INT32:
                return reduce_by_key_from_sorted_vec_dispatch_red<E, K, int32_t>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case INT8:
                return reduce_by_key_from_sorted_vec_dispatch_red<E, K, int8_t>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case FLOAT64:
                return reduce_by_key_from_sorted_vec_dispatch_red<E, K, double>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case FLOAT32:
                return reduce_by_key_from_sorted_vec_dispatch_red<E, K, float>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
        }

        throw std::runtime_error("Invalid dtype for value passed to reduce_by_key_from_sorted_vec");
    }

    template <typename E>
    std::pair<maelstrom::vector, maelstrom::vector> reduce_by_key_from_sorted_vec_dispatch_key(E exec_policy, maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, size_t max_unique_keys) {
        switch(input_keys.get_dtype().prim_type) {
            case UINT64:
                return reduce_by_key_from_sorted_vec_dispatch_val<E, uint64_t>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case UINT32:
                return reduce_by_key_from_sorted_vec_dispatch_val<E, uint32_t>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case UINT8:
                return reduce_by_key_from_sorted_vec_dispatch_val<E, uint8_t>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case INT64:
                return reduce_by_key_from_sorted_vec_dispatch_val<E, int64_t>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case INT32:
                return reduce_by_key_from_sorted_vec_dispatch_val<E, int32_t>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case INT8:
                return reduce_by_key_from_sorted_vec_dispatch_val<E, int8_t>(
                    exec_policy,
                    input_keys,
                    input_values,
                    red,
                    max_unique_keys
                );
            case FLOAT64:
            case FLOAT32:
                throw std::runtime_error("Floating-point types are not supported as keys (error thrown from reduce_by_key_from_sorted_vec)");
        }

        throw std::runtime_error("Invalid dtype for key passed to reduce_by_key_from_sorted_vec");
    }

    std::pair<maelstrom::vector, maelstrom::vector> reduce_by_key_from_sorted_vec_dispatch_exec_policy(maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, size_t max_unique_keys) {
        boost::any exec_policy = maelstrom::get_execution_policy(input_keys).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return reduce_by_key_from_sorted_vec_dispatch_key(
                boost::any_cast<device_exec_t>(exec_policy),
                input_keys,
                input_values,
                red,
                max_unique_keys
            );
        } else if(typeid(host_exec_t) == t) {
            return reduce_by_key_from_sorted_vec_dispatch_key(
                boost::any_cast<host_exec_t>(exec_policy),
                input_keys,
                input_values,
                red,
                max_unique_keys
            );
        }

        throw std::runtime_error("Invalid execution policy");
    }

    std::pair<maelstrom::vector, maelstrom::vector> reduce_by_key(maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, size_t max_unique_keys, bool sorted) {
        if(input_keys.size() != input_values.size()) throw std::runtime_error("Key array size must match value array size in reduce by key");

        if(sorted) {
            return reduce_by_key_from_sorted_vec_dispatch_exec_policy(input_keys, input_values, red, max_unique_keys);
        }

        maelstrom::vector input_keys_copy(input_keys);

        auto ix = maelstrom::sort(input_keys_copy);
        maelstrom::vector input_values_copy = maelstrom::select(input_values, ix);
        
        maelstrom::vector result_vals;
        maelstrom::vector result_ix;

        std::tie(result_vals, result_ix) = reduce_by_key_from_sorted_vec_dispatch_exec_policy(
            input_keys_copy,
            input_values_copy,
            red,
            max_unique_keys
        );

        result_ix = maelstrom::select(ix, result_ix);

        return std::make_pair(
            std::move(result_vals),
            std::move(result_ix)
        );
    }

}