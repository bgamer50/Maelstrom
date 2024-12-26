#include "maelstrom/algorithms/set.h"
#include "maelstrom/thrust_utils/thrust_utils.cuh"
#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/util/any_utils.h"

namespace maelstrom {

    template<typename E, typename T>
    void t_set(E exec_policy, maelstrom::vector& vec, size_t start, size_t end, std::any val) {
        T set_val = std::any_cast<T>(maelstrom::safe_any_cast(val, vec.get_dtype()));

        thrust::fill(
            exec_policy,
            maelstrom::device_tptr_cast<T>(vec.data()) + start,
            maelstrom::device_tptr_cast<T>(vec.data()) + end,
            set_val
        );
    }

    template <typename E>
    void set_dispatch_val(E exec_policy, maelstrom::vector& vec, size_t start, size_t end, std::any val) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return t_set<E, uint64_t>(exec_policy, vec, start, end, val);
            case UINT32:
                return t_set<E, uint32_t>(exec_policy, vec, start, end, val);
            case UINT8:
                return t_set<E, uint8_t>(exec_policy, vec, start, end, val);
            case INT64:
                return t_set<E, int64_t>(exec_policy, vec, start, end, val);
            case INT32:
                return t_set<E, int32_t>(exec_policy, vec, start, end, val);
            case INT8:
                return t_set<E, int8_t>(exec_policy, vec, start, end, val);
            case FLOAT64:
                return t_set<E, double>(exec_policy, vec, start, end, val);
            case FLOAT32:
                return t_set<E, float>(exec_policy, vec, start, end, val);
        }

        throw std::runtime_error("invalid primitive type provided to set");
    }

    void set_dispatch_exec_policy(maelstrom::vector& vec, size_t start, size_t end, std::any val) {
        std::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return set_dispatch_val(
                std::any_cast<device_exec_t>(exec_policy),
                vec,
                start,
                end,
                val
            );
        } else if(typeid(host_exec_t) == t) {
            return set_dispatch_val(
                std::any_cast<host_exec_t>(exec_policy),
                vec,
                start,
                end,
                val
            );
        }

        throw std::runtime_error("Invalid execution policy for set");
    }
 
    void set(maelstrom::vector& vec, size_t start, size_t end, std::any val) {
        if(start > vec.local_size() - 1) throw std::runtime_error("Start out of range!");
        if(end > vec.local_size()) throw std::runtime_error("End out of range!");

        return set_dispatch_exec_policy(vec, start, end, val);
    }

}