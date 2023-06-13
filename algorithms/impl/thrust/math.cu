#include "algorithms/math.h"
#include "thrust_utils/thrust_utils.cuh"
#include "thrust_utils/execution.cuh"

namespace maelstrom {

    template<typename E, typename T>
    maelstrom::vector t_math_binary_op(E exec_policy, maelstrom::vector& left, maelstrom::vector& right, maelstrom::binary_op op) {
        maelstrom::vector output_values(left.get_mem_type(), left.get_dtype(), left.size());

        auto zip_input = thrust::make_zip_iterator(
            maelstrom::device_tptr_cast<T>(left.data()),
            maelstrom::device_tptr_cast<T>(right.data())
        );

        switch(op) {
            case PLUS:
                thrust::transform(
                    zip_input,
                    zip_input + left.size(),
                    maelstrom::device_tptr_cast<T>(output_values.data()),
                    maelstrom::math_binary_plus<T>()
                );
                break;
            case MINUS:
                thrust::transform(
                    zip_input,
                    zip_input + left.size(),
                    maelstrom::device_tptr_cast<T>(output_values.data()),
                    maelstrom::math_binary_minus<T>()
                );
                break;
            case TIMES:
                thrust::transform(
                    zip_input,
                    zip_input + left.size(),
                    maelstrom::device_tptr_cast<T>(output_values.data()),
                    maelstrom::math_binary_times<T>()
                );
                break;
            case DIVIDE:
                thrust::transform(
                    zip_input,
                    zip_input + left.size(),
                    maelstrom::device_tptr_cast<T>(output_values.data()),
                    maelstrom::math_binary_divide<T>()
                );
                break;
            default:
                throw std::runtime_error("Invalid op provided to math_binary_op");
        }

        return output_values;
    }

    template<typename E>
    maelstrom::vector math_binary_op_dispatch_val(E exec_policy, maelstrom::vector& left, maelstrom::vector& right, maelstrom::binary_op op) {
        switch(left.get_dtype().prim_type) {
            case UINT64:
                return t_math_binary_op<E, uint64_t>(
                    exec_policy,
                    left,
                    right,
                    op
                );
            case UINT32:
                return t_math_binary_op<E, uint32_t>(
                    exec_policy,
                    left,
                    right,
                    op
                );
            case UINT8:
                return t_math_binary_op<E, uint8_t>(
                    exec_policy,
                    left,
                    right,
                    op
                );
            case INT64:
                return t_math_binary_op<E, int64_t>(
                    exec_policy,
                    left,
                    right,
                    op
                );
            case INT32:
                return t_math_binary_op<E, int32_t>(
                    exec_policy,
                    left,
                    right,
                    op
                );
            case INT8:
                return t_math_binary_op<E, int8_t>(
                    exec_policy,
                    left,
                    right,
                    op
                );
            case FLOAT64:
                return t_math_binary_op<E, double>(
                    exec_policy,
                    left,
                    right,
                    op
                );
            case FLOAT32:
                return t_math_binary_op<E, float>(
                    exec_policy,
                    left,
                    right,
                    op
                );
        }

        throw std::runtime_error("Invalid dtype provided to math_binary_op");
    }

    maelstrom::vector math_binary_op_dispatch_exec_policy(maelstrom::vector& left, maelstrom::vector& right, maelstrom::binary_op op) {
        boost::any exec_policy = maelstrom::get_execution_policy(left).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return math_binary_op_dispatch_val(
                boost::any_cast<device_exec_t>(exec_policy),
                left,
                right,
                op
            );
        } else if(typeid(host_exec_t) == t) {
            return math_binary_op_dispatch_val(
                boost::any_cast<host_exec_t>(exec_policy),
                left,
                right,
                op
            );
        }

        throw std::runtime_error("Invalid execution policy");
    }

    maelstrom::vector math(maelstrom::vector& left, maelstrom::vector& right, maelstrom::binary_op op) {
        if(left.get_dtype() != right.get_dtype()) throw std::runtime_error("dtype mismatch when calling math");
        if(left.get_mem_type() != right.get_mem_type()) throw std::runtime_error("mem type mismatch when calling math");

        return math_binary_op_dispatch_exec_policy(left, right, op);
    }
}