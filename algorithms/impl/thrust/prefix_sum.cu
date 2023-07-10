#include "maelstrom/algorithms/prefix_sum.h"
#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    template <typename E, typename T>
    void t_prefix_sum(E exec_policy, maelstrom::vector& vec) {
        thrust::inclusive_scan(
            maelstrom::device_tptr_cast<T>(vec.data()),
            maelstrom::device_tptr_cast<T>(vec.data()) + vec.size(),
            maelstrom::device_tptr_cast<T>(vec.data())
        );
    }

    template <typename E>
    void prefix_sum_dispatch_val(E exec_policy, maelstrom::vector& vec) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return t_prefix_sum<E, uint64_t>(exec_policy, vec);
            case UINT32:
                return t_prefix_sum<E, uint32_t>(exec_policy, vec);
            case UINT8:
                return t_prefix_sum<E, uint8_t>(exec_policy, vec);
            case INT64:
                return t_prefix_sum<E, int64_t>(exec_policy, vec);
            case INT32:
                return t_prefix_sum<E, int32_t>(exec_policy, vec);
            case INT8:
                return t_prefix_sum<E, int8_t>(exec_policy, vec);
            case FLOAT64:
                return t_prefix_sum<E, double>(exec_policy, vec);
            case FLOAT32:
                return t_prefix_sum<E, float>(exec_policy, vec);
        }
    }

    void prefix_sum_dispatch_exec_policy(maelstrom::vector& vec) {
        boost::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return prefix_sum_dispatch_val(
                boost::any_cast<device_exec_t>(exec_policy),
                vec
            );
        } else if(typeid(host_exec_t) == t) {
            return prefix_sum_dispatch_val(
                boost::any_cast<host_exec_t>(exec_policy),
                vec
            );
        }

        throw std::runtime_error("Invalid execution policy for increment");
    }

    void prefix_sum(maelstrom::vector& vec) {
        return prefix_sum_dispatch_exec_policy(vec);
    }
}