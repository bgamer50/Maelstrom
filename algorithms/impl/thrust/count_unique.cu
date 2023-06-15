#include "maelstrom/algorithms/count_unique.h"
#include "maelstrom/algorithms/sort.h"

#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    template <typename E, typename T>
    std::pair<maelstrom::vector, maelstrom::vector> t_count_unique_from_sorted_vec(E exec_policy, maelstrom::vector& vec, size_t max_num_values) {
        maelstrom::vector keys_output(
            vec.get_mem_type(),
            vec.get_dtype(),
            max_num_values
        );

        maelstrom::vector values_output(
            vec.get_mem_type(),
            maelstrom::uint64,
            max_num_values
        );

        auto z = thrust::reduce_by_key(
            exec_policy,
            maelstrom::device_tptr_cast<T>(vec.data()),
            maelstrom::device_tptr_cast<T>(vec.data()) + vec.size(),
            thrust::make_constant_iterator<size_t>(static_cast<size_t>(1)),
            maelstrom::device_tptr_cast<T>(keys_output.data()),
            maelstrom::device_tptr_cast<size_t>(values_output.data()),
            thrust::equal_to<T>(),
            thrust::plus<size_t>()
        );

        size_t num_unique = static_cast<size_t>(z.first - maelstrom::device_tptr_cast<T>(keys_output.data()));
        keys_output.resize(num_unique);
        values_output.resize(num_unique);

        if(keys_output.size() < 0.6 * max_num_values) {
            keys_output.shrink_to_fit();
            values_output.shrink_to_fit();
        }

        return std::make_pair(
            std::move(keys_output),
            std::move(values_output)
        );
    }

    template <typename E>
    std::pair<maelstrom::vector, maelstrom::vector> count_unique_from_sorted_vec_dispatch_val(E exec_policy, maelstrom::vector& vec, size_t max_num_values) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return t_count_unique_from_sorted_vec<E, uint64_t>(exec_policy, vec, max_num_values);
            case UINT32:
                return t_count_unique_from_sorted_vec<E, uint32_t>(exec_policy, vec, max_num_values);
            case UINT8:
                return t_count_unique_from_sorted_vec<E, uint8_t>(exec_policy, vec, max_num_values);
            case INT64:
                return t_count_unique_from_sorted_vec<E, int64_t>(exec_policy, vec, max_num_values);
            case INT32:
                return t_count_unique_from_sorted_vec<E, int32_t>(exec_policy, vec, max_num_values);
            case INT8:
                return t_count_unique_from_sorted_vec<E, int8_t>(exec_policy, vec, max_num_values);
            case FLOAT64:
                return t_count_unique_from_sorted_vec<E, double>(exec_policy, vec, max_num_values);
            case FLOAT32:
                return t_count_unique_from_sorted_vec<E, float>(exec_policy, vec, max_num_values);
        }

        throw std::runtime_error("Invalid dtype provided to count_unique_from_sorted_vec()");
    }

    std::pair<maelstrom::vector, maelstrom::vector> count_unique_from_sorted_vec_dispatch_exec_policy(maelstrom::vector& vec, size_t max_num_values) {
        boost::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return count_unique_from_sorted_vec_dispatch_val(
                boost::any_cast<device_exec_t>(exec_policy),
                vec,
                max_num_values
            );
        } else if(typeid(host_exec_t) == t) {
            return count_unique_from_sorted_vec_dispatch_val(
                boost::any_cast<host_exec_t>(exec_policy),
                vec,
                max_num_values
            );
        }

        throw std::runtime_error("Invalid execution policy for count unique");
    }

    std::pair<maelstrom::vector, maelstrom::vector> count_unique(maelstrom::vector& vec, size_t max_num_values, bool sorted) {
        if(!sorted) {
            maelstrom::vector vec_copy(vec);
            maelstrom::sort(vec_copy);

            return maelstrom::count_unique_from_sorted_vec_dispatch_exec_policy(vec_copy, max_num_values);
        }
        
        return maelstrom::count_unique_from_sorted_vec_dispatch_exec_policy(vec, max_num_values);
    }
}