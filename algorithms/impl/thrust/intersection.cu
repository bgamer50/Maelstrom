#include "algorithms/intersection.h"
#include "algorithms/sort.h"
#include "algorithms/select.h"

#include "thrust_utils/thrust_utils.cuh"
#include "thrust_utils/execution.cuh"

namespace maelstrom {

    template<typename T>
    struct intersection_less_cmp {
        __host__ __device__ bool operator() (const thrust::tuple<T, size_t>& left, const thrust::tuple<T, size_t>& right) {
            return thrust::get<0>(left) < thrust::get<0>(right);
        }
    };

    template <typename E, typename T>
    maelstrom::vector t_intersection(E exec_policy, maelstrom::vector& left, maelstrom::vector& right) {
        auto zip_left = thrust::make_zip_iterator(
            maelstrom::device_tptr_cast<T>(left.data()),
            thrust::make_counting_iterator(static_cast<size_t>(0))
        );

        auto zip_right = thrust::make_zip_iterator(
            maelstrom::device_tptr_cast<T>(right.data()),
            thrust::make_counting_iterator(static_cast<size_t>(0))
        );

        maelstrom::vector output_indices(
            left.get_mem_type(),
            uint64,
            (left.size() > right.size()) ? right.size() : left.size()
        );

        auto zip_output = thrust::make_zip_iterator(
            thrust::make_discard_iterator(),
            maelstrom::device_tptr_cast<size_t>(output_indices.data())
        );

        auto end = thrust::set_intersection(
            exec_policy,
            zip_left,
            zip_left + left.size(),
            zip_right,
            zip_right + right.size(),
            zip_output,
            intersection_less_cmp<T>()
        );

        size_t new_size = static_cast<size_t>(end - zip_output);
        output_indices.resize(new_size);
        output_indices.shrink_to_fit();

        return output_indices;
    }

    template <typename E>
    maelstrom::vector intersection_dispatch_val(E exec_policy, maelstrom::vector& left, maelstrom::vector& right) {
        switch(left.get_dtype().prim_type) {
            case UINT64:
                return t_intersection<E, uint64_t>(exec_policy, left, right);
            case UINT32:
                return t_intersection<E, uint32_t>(exec_policy, left, right);
            case UINT8:
                return t_intersection<E, uint8_t>(exec_policy, left, right);
            case INT64:
                return t_intersection<E, int64_t>(exec_policy, left, right);
            case INT32:
                return t_intersection<E, int32_t>(exec_policy, left, right);
            case INT8:
                return t_intersection<E, int8_t>(exec_policy, left, right);
            case FLOAT64:
                return t_intersection<E, double>(exec_policy, left, right);
            case FLOAT32:
                return t_intersection<E, float>(exec_policy, left, right);
        }

        throw std::runtime_error("invalid dtype for intersection");
    }

    maelstrom::vector intersection_dispatch_exec_policy(maelstrom::vector& left, maelstrom::vector& right) {
        boost::any exec_policy = maelstrom::get_execution_policy(left).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return intersection_dispatch_val(
                boost::any_cast<device_exec_t>(exec_policy),
                left,
                right
            );
        } else if(typeid(host_exec_t) == t) {
            return intersection_dispatch_val(
                boost::any_cast<host_exec_t>(exec_policy),
                left,
                right
            );
        }

        throw std::runtime_error("Invalid execution policy");
    }

    maelstrom::vector intersection(maelstrom::vector& left, maelstrom::vector& right, bool sorted) {
        if(left.get_mem_type() != right.get_mem_type()) throw std::runtime_error("left mem type must match right mem type for intersection!");
        if(left.get_dtype() != right.get_dtype()) throw std::runtime_error("left dtype must match right dtype for intersection");

        if(sorted) return intersection_dispatch_exec_policy(left, right);

        maelstrom::vector left_copy(left);
        auto left_ix = maelstrom::sort(left_copy);

        maelstrom::vector right_copy(right);
        maelstrom::sort(right_copy);

        auto int_ix = intersection_dispatch_exec_policy(left_copy, right_copy);
        return maelstrom::select(left_ix, int_ix);
    }

}