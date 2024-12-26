#include "maelstrom/algorithms/topk.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/arange.h"

#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    template <typename T>
    struct topk_binary_max : public thrust::binary_function<thrust::tuple<T, size_t>, thrust::tuple<T, size_t>, thrust::tuple<T, size_t>> {
        size_t* ix;
        size_t invalid;
        bool descending;
        __device__ __host__ thrust::tuple<T, size_t> operator()(const thrust::tuple<T, size_t>& left, const thrust::tuple<T, size_t>& right) {
            if(thrust::get<1>(left) == invalid) return right;
            if(thrust::get<1>(right) == invalid) return left;

            for(size_t i = 0; ix[i] != invalid; ++i) {
                if(ix[i] == thrust::get<1>(left)) return right;
                if(ix[i] == thrust::get<1>(right)) return left;
            }

            bool cmp = thrust::get<0>(left) < thrust::get<0>(right);
            return cmp ? (descending ? left : right) : (descending ? right : left);
        }
    };

    template <typename E, typename T>
    maelstrom::vector t_topk(E exec_policy, maelstrom::vector& vec, size_t k, bool descending) {
        const size_t invalid = vec.size();
        maelstrom::vector topk_vec(vec.get_mem_type(), maelstrom::uint64, k);
        maelstrom::set(topk_vec, invalid);

        topk_binary_max<T> op;
        op.ix = static_cast<size_t*>(topk_vec.data());
        op.invalid = invalid;
        op.descending = descending;

        auto zip = thrust::make_zip_iterator(
            maelstrom::device_tptr_cast<T>(vec.data()),
            thrust::make_counting_iterator(static_cast<size_t>(0))
        );

        T start_val = descending ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();

        for(size_t q = 0; q < k; ++q) {
            auto i_t = thrust::reduce(
                exec_policy,
                zip,
                zip + vec.size(),
                thrust::make_tuple<T, size_t>(start_val, invalid),
                op
            );
            size_t i = thrust::get<1>(i_t);

            // FIXME this bends the rules of expected behavior for a Maelstrom algorithm
            auto str = std::any_cast<cudaStream_t>(vec.get_stream());
            cudaMemcpyAsync(static_cast<size_t*>(topk_vec.data()) + q, &i, sizeof(size_t) * 1, cudaMemcpyDefault, str);
            cudaStreamSynchronize(str);
        }

        return topk_vec;
    }

    template <typename E>
    maelstrom::vector topk_dispatch_val(E exec_policy, maelstrom::vector& vec, size_t k, bool descending) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return t_topk<E, uint64_t>(exec_policy, vec, k, descending);
            case UINT32:
                return t_topk<E, uint32_t>(exec_policy, vec, k, descending);
            case UINT8:
                return t_topk<E, uint8_t>(exec_policy, vec, k, descending);
            case INT64:
                return t_topk<E, int64_t>(exec_policy, vec, k, descending);
            case INT32:
                return t_topk<E, int32_t>(exec_policy, vec, k, descending);
            case INT8:
                return t_topk<E, int8_t>(exec_policy, vec, k, descending);
            case FLOAT64:
                return t_topk<E, double>(exec_policy, vec, k, descending);
            case FLOAT32:
                return t_topk<E, float>(exec_policy, vec, k, descending);
        }

        throw std::runtime_error("invalid primitive type provided to set");
    }

    maelstrom::vector topk_dispatch_exec_policy(maelstrom::vector& vec, size_t k, bool descending) {
        std::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return topk_dispatch_val(
                std::any_cast<device_exec_t>(exec_policy),
                vec,
                k,
                descending
            );
        } else if(typeid(host_exec_t) == t) {
            return topk_dispatch_val(
                std::any_cast<host_exec_t>(exec_policy),
                vec,
                k,
                descending
            );
        }

        throw std::runtime_error("Invalid execution policy for topk");
    }

    maelstrom::vector topk(maelstrom::vector& vec, size_t k, bool descending) {
        return topk_dispatch_exec_policy(vec, k < vec.size() ? k : vec.size(), descending);
    }

}