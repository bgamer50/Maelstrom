#include "algorithms/cast.h"
#include "thrust_utils/thrust_utils.cuh"
#include "thrust_utils/execution.cuh"

namespace maelstrom {

    template <typename E, typename I, typename O>
    maelstrom::vector t_cast(E exec_policy, maelstrom::vector& vec, maelstrom::dtype_t new_type) {
        maelstrom::vector new_vec(vec.get_mem_type(), new_type, vec.size());

        thrust::transform(
            exec_policy,
            maelstrom::device_tptr_cast<I>(vec.data()),
            maelstrom::device_tptr_cast<I>(vec.data()) + vec.size(),
            maelstrom::device_tptr_cast<O>(new_vec.data()),
            maelstrom::cast_fn<I, O>()
        );

        return new_vec;
    }

    template <typename E, typename I>
    maelstrom::vector cast_dispatch_new_type(E exec_policy, maelstrom::vector& vec, maelstrom::dtype_t new_type) {
        switch(new_type.prim_type) {
            case UINT64:
                return t_cast<E, I, uint64_t>(exec_policy, vec, new_type);
            case UINT32:
                return t_cast<E, I, uint32_t>(exec_policy, vec, new_type);
            case UINT8:
                return t_cast<E, I, uint64_t>(exec_policy, vec, new_type);
            case INT64:
                return t_cast<E, I, int64_t>(exec_policy, vec, new_type);
            case INT32:
                return t_cast<E, I, int32_t>(exec_policy, vec, new_type);
            case INT8:
                return t_cast<E, I, int8_t>(exec_policy, vec, new_type);
            case FLOAT64:
                return t_cast<E, I, double>(exec_policy, vec, new_type);
            case FLOAT32:
                return t_cast<E, I, float>(exec_policy, vec, new_type);
        }

        throw std::runtime_error("Invalid dtype provided to cast");
    }

    template <typename E>
    maelstrom::vector cast_dispatch_old_type(E exec_policy, maelstrom::vector& vec, maelstrom::dtype_t new_type) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return cast_dispatch_new_type<E, uint64_t>(exec_policy, vec, new_type);
            case UINT32:
                return cast_dispatch_new_type<E, uint32_t>(exec_policy, vec, new_type);
            case UINT8:
                return cast_dispatch_new_type<E, uint8_t>(exec_policy, vec, new_type);
            case INT64:
                return cast_dispatch_new_type<E, int64_t>(exec_policy, vec, new_type);
            case INT32:
                return cast_dispatch_new_type<E, int32_t>(exec_policy, vec, new_type);
            case INT8:
                return cast_dispatch_new_type<E, int8_t>(exec_policy, vec, new_type);
            case FLOAT64:
                return cast_dispatch_new_type<E, double>(exec_policy, vec, new_type);
            case FLOAT32:
                return cast_dispatch_new_type<E, float>(exec_policy, vec, new_type);
        }

        throw std::runtime_error("Invalid dtype provided to cast");
    }

    maelstrom::vector cast_dispatch_exec_policy(maelstrom::vector& vec, maelstrom::dtype_t new_type) {
        boost::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return cast_dispatch_old_type(
                boost::any_cast<device_exec_t>(exec_policy),
                vec,
                new_type
            );
        } else if(typeid(host_exec_t) == t) {
            return cast_dispatch_old_type(
                boost::any_cast<host_exec_t>(exec_policy),
                vec,
                new_type
            );
        }

        throw std::runtime_error("Invalid execution policy");
    }

    maelstrom::vector cast(maelstrom::vector& vec, maelstrom::dtype_t new_type) {
        if(vec.get_dtype() == new_type) return maelstrom::vector(vec);
        
        return cast_dispatch_exec_policy(vec, new_type);
    }
}