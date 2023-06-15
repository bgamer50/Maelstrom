#include "maelstrom/algorithms/sort.h"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    template<typename E, typename T>
    maelstrom::vector t_sort(E exec_policy, maelstrom::vector& vec) {
        maelstrom::vector sorted_indices(
            vec.get_mem_type(),
            uint64,
            vec.size()
        );

        thrust::copy(
            exec_policy,
            thrust::make_counting_iterator(static_cast<size_t>(0)),
            thrust::make_counting_iterator(static_cast<size_t>(0)) + vec.size(),
            maelstrom::device_tptr_cast<size_t>(sorted_indices.data())
        );

        thrust::sort_by_key(
            exec_policy,
            maelstrom::device_tptr_cast<T>(vec.data()),
            maelstrom::device_tptr_cast<T>(vec.data()) + vec.size(),
            maelstrom::device_tptr_cast<size_t>(sorted_indices.data())
        );

        return sorted_indices;
    }

    template <typename E>
    maelstrom::vector sort_dispatch_val(E exec_policy, maelstrom::vector& vec) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return t_sort<E, uint64_t>(exec_policy, vec);
            case UINT32:
                return t_sort<E, uint32_t>(exec_policy, vec);
            case UINT8:
                return t_sort<E, uint8_t>(exec_policy, vec);
            case INT64:
                return t_sort<E, int64_t>(exec_policy, vec);
            case INT32:
                return t_sort<E, int32_t>(exec_policy, vec);
            case INT8:
                return t_sort<E, int8_t>(exec_policy, vec);
            case FLOAT64:
                return t_sort<E, double>(exec_policy, vec);
            case FLOAT32:
                return t_sort<E, float>(exec_policy, vec);
        }

        throw std::runtime_error("Invalid dtype provided to sort()");
    }

    maelstrom::vector sort_dispatch_exec_policy(maelstrom::vector& vec) {
        boost::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return sort_dispatch_val(
                boost::any_cast<device_exec_t>(exec_policy),
                vec
            );
        } else if(typeid(host_exec_t) == t) {
            return sort_dispatch_val(
                boost::any_cast<host_exec_t>(exec_policy),
                vec
            );
        }

        throw std::runtime_error("Invalid execution policy for sort");
    }

    maelstrom::vector sort(maelstrom::vector& vec) {
        return sort_dispatch_exec_policy(vec);
    }

}