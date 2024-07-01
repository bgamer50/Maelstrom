#include "maelstrom/algorithms/reverse.h"
#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    template<typename E, typename T>
    void t_reverse(E exec_policy, maelstrom::vector& vec) {
        thrust::reverse(
            exec_policy,
            maelstrom::device_tptr_cast<T>(vec.data()),
            maelstrom::device_tptr_cast<T>(vec.data()) + vec.size()
        );
    }

    template<typename E>
    void reverse_dispatch_val(E exec_policy, maelstrom::vector& vec) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return t_reverse<E, uint64_t>(exec_policy, vec);
            case UINT32:
                return t_reverse<E, uint32_t>(exec_policy, vec);
            case UINT8:
                return t_reverse<E, uint8_t>(exec_policy, vec);
            case INT64:
                return t_reverse<E, int64_t>(exec_policy, vec);
            case INT32:
                return t_reverse<E, int32_t>(exec_policy, vec);
            case INT8:
                return t_reverse<E, int8_t>(exec_policy, vec);
            case FLOAT64:
                return t_reverse<E, double>(exec_policy, vec);
            case FLOAT32:
                return t_reverse<E, float>(exec_policy, vec);
        }

        throw std::invalid_argument("Invalid data type for reverse");
    }

    void reverse_dispatch_exec_policy(maelstrom::vector& vec) {
        std::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return reverse_dispatch_val(
                std::any_cast<device_exec_t>(exec_policy),
                vec
            );
        } else if(typeid(host_exec_t) == t) {
            return reverse_dispatch_val(
                std::any_cast<host_exec_t>(exec_policy),
                vec
            );
        }

        throw std::runtime_error("Invalid execution policy for filter");
    }

    void reverse(maelstrom::vector& vec) {
        return reverse_dispatch_exec_policy(vec);
    }

}