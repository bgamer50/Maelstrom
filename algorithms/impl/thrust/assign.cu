#include "maelstrom/algorithms/assign.h"
#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    extern void assign_dispatch_dist(maelstrom::vector& dst, maelstrom::vector& ix, maelstrom::vector& values);

    template <typename E, typename I, typename V>
    void t_assign(E exec_policy, maelstrom::vector& dst, maelstrom::vector& ix, maelstrom::vector& values) {
        thrust::scatter(
            exec_policy,
            maelstrom::device_tptr_cast<V>(values.data()),
            maelstrom::device_tptr_cast<V>(values.data()) + values.size(),
            maelstrom::device_tptr_cast<I>(ix.data()),
            maelstrom::device_tptr_cast<V>(dst.data())
        );
    }

    template<typename E, typename I>
    void assign_dispatch_val(E exec_policy, maelstrom::vector& dst, maelstrom::vector& ix, maelstrom::vector& values) {
        switch(dst.get_dtype().prim_type) {
            case UINT64:
                return t_assign<E, I, uint64_t>(exec_policy, dst, ix, values);
            case UINT32:
                return t_assign<E, I, uint32_t>(exec_policy, dst, ix, values);
            case UINT8:
                return t_assign<E, I, uint8_t>(exec_policy, dst, ix, values);
            case INT64:
                return t_assign<E, I, int64_t>(exec_policy, dst, ix, values);
            case INT32:
                return t_assign<E, I, int32_t>(exec_policy, dst, ix, values);
            case INT8:
                return t_assign<E, I, int8_t>(exec_policy, dst, ix, values);
            case FLOAT64:
                return t_assign<E, I, double>(exec_policy, dst, ix, values);
            case FLOAT32:
                return t_assign<E, I, float>(exec_policy, dst, ix, values);
        }
    }

    template<typename E>
    void assign_dispatch_index(E exec_policy, maelstrom::vector& dst, maelstrom::vector& ix, maelstrom::vector& values) {
        switch(ix.get_dtype().prim_type) {
            case UINT64:
                return assign_dispatch_val<E, uint64_t>(exec_policy, dst, ix, values);
            case UINT32:
                return assign_dispatch_val<E, uint32_t>(exec_policy, dst, ix, values);
            case INT64:
                return assign_dispatch_val<E, int64_t>(exec_policy, dst, ix, values);
            case INT32:
                return assign_dispatch_val<E, int32_t>(exec_policy, dst, ix, values);
        }

        throw std::runtime_error("invalid index type for assign");
    }

    void assign_dispatch_exec_policy(maelstrom::vector& dst, maelstrom::vector& ix, maelstrom::vector& values) {
        std::any exec_policy = maelstrom::get_execution_policy(dst).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return assign_dispatch_index(
                std::any_cast<device_exec_t>(exec_policy),
                dst,
                ix,
                values
            );
        } else if(typeid(host_exec_t) == t) {
            return assign_dispatch_index(
                std::any_cast<host_exec_t>(exec_policy),
                dst,
                ix,
                values
            );
        }

        throw std::runtime_error("Invalid execution policy for assign");
    }

    void assign(maelstrom::vector& dst, maelstrom::vector& ix, maelstrom::vector& values) {
        if(dst.get_dtype() != values.get_dtype()) throw std::runtime_error("values dtype does not match destination vector dtype");
        if(ix.size() != values.size()) throw std::runtime_error("index size must match values size");
        
        auto mem_type = dst.get_mem_type();
        if(maelstrom::is_dist(mem_type)) {
            return assign_dispatch_dist(dst, ix, values);
        }

        return assign_dispatch_exec_policy(dst, ix, values);
    }

}