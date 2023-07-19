#include "maelstrom/algorithms/filter.h"
#include "maelstrom/thrust_utils/thrust_utils.cuh"
#include "maelstrom/thrust_utils/thrust_comparison.cuh"

#include <sstream>

namespace maelstrom {

    template <typename E, typename T>
    maelstrom::vector t_filter(E exec_policy, maelstrom::vector& vec, maelstrom::comparator cmp, std::any cmp_val) {
        maelstrom::vector indices(
            vec.get_mem_type(),
            maelstrom::uint64,
            vec.size()
        );

        if(cmp == maelstrom::comparator::BETWEEN || cmp == maelstrom::comparator::INSIDE || cmp == maelstrom::comparator::OUTSIDE || cmp == maelstrom::comparator::IS_IN || cmp == maelstrom::comparator::IS_NOT_IN) {
            std::stringstream sx;
            sx << "Comparator " << cmp << " is currently unsupported for filter()";
            throw std::runtime_error(sx.str());
        } else {
            if(maelstrom::prim_type_of(cmp_val) != vec.get_dtype().prim_type) throw std::runtime_error("Type of cmp value does not match vector primitive type");
        }

        auto end = thrust::copy_if(
            exec_policy,
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(0) + vec.size(),
            maelstrom::device_tptr_cast<T>(vec.data()),
            maelstrom::device_tptr_cast<size_t>(indices.data()),
            maelstrom::filter_fn<T>{std::any_cast<T>(cmp_val), cmp}
        );

        indices.resize(
            static_cast<size_t>(end - maelstrom::device_tptr_cast<size_t>(indices.data()))
        );

        return indices;
    }

    template <typename E>
    maelstrom::vector filter_dispatch_val(E exec_policy, maelstrom::vector& vec, maelstrom::comparator cmp, std::any cmp_val) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return t_filter<E, uint64_t>(exec_policy, vec, cmp, cmp_val);
            case UINT32:
                return t_filter<E, uint32_t>(exec_policy, vec, cmp, cmp_val);
            case UINT8:
                return t_filter<E, uint8_t>(exec_policy, vec, cmp, cmp_val);
            case INT64:
                return t_filter<E, int64_t>(exec_policy, vec, cmp, cmp_val);
            case INT32:
                return t_filter<E, int32_t>(exec_policy, vec, cmp, cmp_val);
            case INT8:
                return t_filter<E, int8_t>(exec_policy, vec, cmp, cmp_val);
            case FLOAT64:
                return t_filter<E, double>(exec_policy, vec, cmp, cmp_val);
            case FLOAT32:
                return t_filter<E, float>(exec_policy, vec, cmp, cmp_val);
        }

        throw std::runtime_error("Invalid dtype provided to filter()");
    }

    maelstrom::vector filter_dispatch_exec_policy(maelstrom::vector& vec, maelstrom::comparator cmp, std::any cmp_val) {
        std::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return filter_dispatch_val(
                std::any_cast<device_exec_t>(exec_policy),
                vec,
                cmp,
                cmp_val
            );
        } else if(typeid(host_exec_t) == t) {
            return filter_dispatch_val(
                std::any_cast<host_exec_t>(exec_policy),
                vec,
                cmp,
                cmp_val
            );
        }

        throw std::runtime_error("Invalid execution policy for filter");
    }

    maelstrom::vector filter(maelstrom::vector& vec, maelstrom::comparator cmp, std::any cmp_val) {
        return filter_dispatch_exec_policy(vec, cmp, cmp_val);
    }

}