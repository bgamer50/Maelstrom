#include "maelstrom/algorithms/increment.h"

#include "maelstrom/thrust_utils/thrust_utils.cuh"
#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/util/any_utils.cuh"

namespace maelstrom {

    template<typename E, typename T>
    void t_increment(E exec_policy, maelstrom::vector& vec, std::any inc, size_t start, size_t end, maelstrom::inc_op op) {
        T inc_val;

        if(!inc.has_value()) {
            inc_val = static_cast<T>(1);
        } else {
            inc_val = std::any_cast<T>(maelstrom::safe_any_cast(inc, vec.get_dtype()));
        }

        maelstrom::unary_plus_op<T> adder;
        adder.plus_val = inc_val;
        
        switch(op) {
            case INCREMENT:
                adder.subtract = false;
                break;
            case DECREMENT:
                adder.subtract = true;
                break;
            default:
                throw std::runtime_error("Invalid increment operation");
        }

        thrust::transform(
            exec_policy,
            device_tptr_cast<T>(vec.data()) + start,
            device_tptr_cast<T>(vec.data()) + end,
            device_tptr_cast<T>(vec.data()) + start,
            adder
        );
    }

    template<typename E>
    void increment_dispatch_val(E exec_policy, maelstrom::vector& vec, std::any inc, size_t start, size_t end, maelstrom::inc_op op) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return t_increment<E, uint64_t>(exec_policy, vec, inc, start, end, op);
            case UINT32:
                return t_increment<E, uint32_t>(exec_policy, vec, inc, start, end, op);
            case UINT8:
                return t_increment<E, uint8_t>(exec_policy, vec, inc, start, end, op);
            case INT64:
                return t_increment<E, int64_t>(exec_policy, vec, inc, start, end, op);
            case INT32:
                return t_increment<E, int32_t>(exec_policy, vec, inc, start, end, op);
            case INT8:
                return t_increment<E, int8_t>(exec_policy, vec, inc, start, end, op);
            case FLOAT64:
                return t_increment<E, double>(exec_policy, vec, inc, start, end, op);
            case FLOAT32:
                return t_increment<E, float>(exec_policy, vec, inc, start, end, op);
        }

        throw std::runtime_error("invalid primitive type provided to increment");
    }

    void increment_dispatch_exec_policy(maelstrom::vector& vec, std::any inc, size_t start, size_t end, maelstrom::inc_op op) {
        std::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return increment_dispatch_val(
                std::any_cast<device_exec_t>(exec_policy),
                vec,
                inc,
                start,
                end,
                op
            );
        } else if(typeid(host_exec_t) == t) {
            return increment_dispatch_val(
                std::any_cast<host_exec_t>(exec_policy),
                vec,
                inc,
                start,
                end,
                op
            );
        }

        throw std::runtime_error("Invalid execution policy for increment");
    }

    void increment(maelstrom::vector& vec, std::any inc, size_t start, size_t end, maelstrom::inc_op op) {
        if(start > vec.size() - 1) throw std::runtime_error("Start out of range!");
        if(end > vec.size()) throw std::runtime_error("End out of range!");

        increment_dispatch_exec_policy(vec, inc, start, end, op);
    }

}