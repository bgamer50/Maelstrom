#include "maelstrom/algorithms/compare.h"
#include "maelstrom/thrust_utils/thrust_utils.cuh"
#include "maelstrom/thrust_utils/thrust_comparison.cuh"

namespace maelstrom {

    template <typename E, typename T>
    maelstrom::vector t_compare(E exec_policy, maelstrom::vector& vec1, maelstrom::vector& vec2, maelstrom::comparator cmp, bool invert) {
        maelstrom::vector output(
            vec1.get_mem_type(),
            uint8,
            vec1.size()
        );

        thrust::transform(
            exec_policy,
            maelstrom::device_tptr_cast<T>(vec1.data()),
            maelstrom::device_tptr_cast<T>(vec1.data()) + vec1.size(),
            maelstrom::device_tptr_cast<T>(vec2.data()),
            maelstrom::device_tptr_cast<uint8_t>(output.data()),
            maelstrom::compare_fn<T>{cmp, invert}
        );

        return output;
    }

    template <typename E>
    maelstrom::vector compare_dispatch_val(E exec_policy, maelstrom::vector& vec1, maelstrom::vector& vec2, maelstrom::comparator cmp, bool invert) {
        switch(vec1.get_dtype().prim_type) {
            case UINT64:
                return t_compare<E, uint64_t>(
                    exec_policy,
                    vec1,
                    vec2,
                    cmp,
                    invert
                );
            case UINT32:
                return t_compare<E, uint32_t>(
                    exec_policy,
                    vec1,
                    vec2,
                    cmp,
                    invert
                );
            case UINT8:
                return t_compare<E, uint8_t>(
                    exec_policy,
                    vec1,
                    vec2,
                    cmp,
                    invert
                );
            case INT64:
                return t_compare<E, int64_t>(
                    exec_policy,
                    vec1,
                    vec2,
                    cmp,
                    invert
                );
            case INT32:
                return t_compare<E, int32_t>(
                    exec_policy,
                    vec1,
                    vec2,
                    cmp,
                    invert
                );
            case INT8:
                return t_compare<E, int8_t>(
                    exec_policy,
                    vec1,
                    vec2,
                    cmp,
                    invert
                );
            case FLOAT64:
                return t_compare<E, double>(
                    exec_policy,
                    vec1,
                    vec2,
                    cmp,
                    invert
                );
            case FLOAT32:
                return t_compare<E, float>(
                    exec_policy,
                    vec1,
                    vec2,
                    cmp,
                    invert
                );
        }

        throw std::runtime_error("Invalid dtype for compare()");
    }

    maelstrom::vector compare_dispatch_exec_policy(maelstrom::vector& vec1, maelstrom::vector& vec2, maelstrom::comparator cmp, bool invert) {
        std::any exec_policy = maelstrom::get_execution_policy(vec1).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return compare_dispatch_val(
                std::any_cast<device_exec_t>(exec_policy),
                vec1,
                vec2,
                cmp,
                invert
            );
        } else if(typeid(host_exec_t) == t) {
            return compare_dispatch_val(
                std::any_cast<host_exec_t>(exec_policy),
                vec1,
                vec2,
                cmp,
                invert
            );
        }

        throw std::runtime_error("Invalid execution policy for compare");

    }

    maelstrom::vector compare(maelstrom::vector& vec1, maelstrom::vector& vec2, maelstrom::comparator cmp, bool invert) {
        if(vec1.get_dtype() != vec2.get_dtype()) {
            throw std::runtime_error("Data types of vectors mismatch.  Cannot call compare()");
        }
        if(vec1.get_mem_type() != vec2.get_mem_type()) {
            throw std::runtime_error("Memory types of vectors mismatch.  Cannot call compare()");
        }
        if(vec1.size() != vec2.size()) {
            throw std::runtime_error("Vector sizes mismatch.  Cannot call compare()");
        }

        return compare_dispatch_exec_policy(vec1, vec2, cmp, invert);
    }

}