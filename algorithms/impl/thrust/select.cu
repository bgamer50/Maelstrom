#include "maelstrom/algorithms/select.h"

#include "maelstrom/thrust_utils/thrust_utils.cuh"
#include "maelstrom/thrust_utils/execution.cuh"

#include <sstream>

namespace maelstrom {
    template<typename E, typename T, typename I>
    maelstrom::vector t_select(E thrust_exec_policy, maelstrom::vector& vec, maelstrom::vector& idx) {
        auto out = maelstrom::make_vector_like(vec);
        out.resize(idx.size());

        thrust::gather(
            thrust_exec_policy,
            maelstrom::device_tptr_cast<I>(idx.data()),
            maelstrom::device_tptr_cast<I>(idx.data()) + idx.size(),
            maelstrom::device_tptr_cast<T>(vec.data()),
            maelstrom::device_tptr_cast<T>(out.data())
        );

        return out;
    }

    template<typename E, typename T>
    maelstrom::vector select_dispatch_inner(E thrust_exec_policy, maelstrom::vector& vec, maelstrom::vector& idx) {
        switch(idx.get_dtype().prim_type) {
            case UINT64:
                return t_select<E, T, uint64_t>(thrust_exec_policy, vec, idx);
            case UINT32:
                return t_select<E, T, uint32_t>(thrust_exec_policy, vec, idx);
            case UINT8:
                return t_select<E, T, uint8_t>(thrust_exec_policy, vec, idx);
            case INT64:
                return t_select<E, T, int64_t>(thrust_exec_policy, vec, idx);
            case INT32:
                return t_select<E, T, int32_t>(thrust_exec_policy, vec, idx);
            case INT8:
                return t_select<E, T, int8_t>(thrust_exec_policy, vec, idx);
        }

        if(vec.get_dtype().prim_type == FLOAT32 || vec.get_dtype().prim_type == FLOAT64) {
            throw std::runtime_error("cannot use floating point type for indexing");
        }

        throw std::runtime_error("invalid dtype provided for index vector provided to select");
    }

    template<typename E>
    maelstrom::vector select_dispatch_outer(E thrust_exec_policy, maelstrom::vector& vec, maelstrom::vector& idx) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return select_dispatch_inner<E, uint64_t>(thrust_exec_policy, vec, idx);
            case UINT32:
                return select_dispatch_inner<E, uint32_t>(thrust_exec_policy, vec, idx);
            case UINT8:
                return select_dispatch_inner<E, uint8_t>(thrust_exec_policy, vec, idx);
            case INT64:
                return select_dispatch_inner<E, int64_t>(thrust_exec_policy, vec, idx);
            case INT32:
                return select_dispatch_inner<E, int32_t>(thrust_exec_policy, vec, idx);
            case INT8:
                return select_dispatch_inner<E, int8_t>(thrust_exec_policy, vec, idx);
            case FLOAT64:
                return select_dispatch_inner<E, double>(thrust_exec_policy, vec, idx);
            case FLOAT32:
                return select_dispatch_inner<E, float>(thrust_exec_policy, vec, idx);
        }

        throw std::runtime_error("Invalid dtype for data vector provided to select!");
    }

    maelstrom::vector select(maelstrom::vector& vec, maelstrom::vector& idx) {
        // Error checking
        // TODO do this properly
        /*
        if(vec.get_mem_type() != idx.get_mem_type()) {
            std::stringstream sx;
            sx << "Memory type of array (" << vec.get_mem_type() << ")";
            sx << " does not match memory type of index (" << idx.get_mem_type() << ")";
            throw std::runtime_error(sx.str());
        }
        */

        std::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return select_dispatch_outer(
                std::any_cast<device_exec_t>(exec_policy),
                vec,
                idx
            );
        } else if(typeid(host_exec_t) == t) {
            return select_dispatch_outer(
                std::any_cast<host_exec_t>(exec_policy),
                vec,
                idx
            );
        }

        throw std::runtime_error("Invalid execution policy for select");
    }
}