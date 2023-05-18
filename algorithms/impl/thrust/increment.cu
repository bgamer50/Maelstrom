#include "algorithms/increment.h"

#include "thrust_utils/thrust_utils.cuh"
#include "thrust_utils/execution.cuh"

namespace maelstrom {

    template<typename E, typename T>
    void t_increment(E exec_policy, maelstrom::vector& vec, boost::any inc, size_t start, size_t end) {
        T inc_val;
        try {
            inc_val = boost::any_cast<T>(inc);
        } catch(boost::bad_any_cast& err) {
            throw std::runtime_error("Type of value does not match type of array in increment()");
        }

        maelstrom::unary_plus_op<T> adder;
        adder.plus_val = inc_val;

        thrust::transform(
            exec_policy,
            device_tptr_cast<T>(vec.data()) + start,
            device_tptr_cast<T>(vec.data()) + end,
            device_tptr_cast<T>(vec.data()) + start,
            adder
        );
    }

    template<typename E>
    void increment_dispatch_val(E exec_policy, maelstrom::vector& vec, boost::any inc, size_t start, size_t end) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return t_increment<E, uint64_t>(exec_policy, vec, inc, start, end);
            case UINT32:
                return t_increment<E, uint32_t>(exec_policy, vec, inc, start, end);
            case UINT8:
                return t_increment<E, uint8_t>(exec_policy, vec, inc, start, end);
            case INT64:
                return t_increment<E, int64_t>(exec_policy, vec, inc, start, end);
            case INT32:
                return t_increment<E, int32_t>(exec_policy, vec, inc, start, end);
            case INT8:
                return t_increment<E, int8_t>(exec_policy, vec, inc, start, end);
            case FLOAT64:
                return t_increment<E, double>(exec_policy, vec, inc, start, end);
            case FLOAT32:
                return t_increment<E, float>(exec_policy, vec, inc, start, end);
        }

        throw std::runtime_error("invalid primitive type provided to increment");
    }

    void increment_dispatch_exec_policy(maelstrom::vector& vec, boost::any inc, size_t start, size_t end) {
        boost::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return increment_dispatch_val(
                boost::any_cast<device_exec_t>(exec_policy),
                vec,
                inc,
                start,
                end
            );
        } else if(typeid(host_exec_t) == t) {
            return increment_dispatch_val(
                boost::any_cast<host_exec_t>(exec_policy),
                vec,
                inc,
                start,
                end
            );
        }

        throw std::runtime_error("Invalid execution policy for increment");
    }

    void increment(maelstrom::vector& vec, boost::any inc, size_t start, size_t end) {
        if(start > vec.size() - 1) throw std::runtime_error("Start out of range!");
        if(end > vec.size()) throw std::runtime_error("End out of range!");

        increment_dispatch_exec_policy(vec, inc, start, end);
    }

}