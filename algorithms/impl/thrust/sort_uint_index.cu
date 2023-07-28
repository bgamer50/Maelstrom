#include "maelstrom/containers/vector.h"
#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    template <typename E, typename T1, typename T2, typename T3>
    maelstrom::vector t_sort_uint_index(E exec_policy, std::vector<std::reference_wrapper<maelstrom::vector>> vectors) {
        maelstrom::vector sorted_indices(
            vectors.front().get().get_mem_type(),
            uint64,
            vectors.front().get().size()
        );

        thrust::copy(
            exec_policy,
            thrust::make_counting_iterator(static_cast<size_t>(0)),
            thrust::make_counting_iterator(static_cast<size_t>(0)) + vectors.front().get().size(),
            maelstrom::device_tptr_cast<size_t>(sorted_indices.data())
        );

        switch(vectors.size()) {
            case 0:
                throw std::runtime_error("no vectors provided to sort");
            case 1: {
                thrust::sort_by_key(
                    exec_policy,
                    maelstrom::device_tptr_cast<T1>(vectors.front().get().data()),
                    maelstrom::device_tptr_cast<T1>(vectors.front().get().data()) + vectors.front().get().size(),
                    maelstrom::device_tptr_cast<size_t>(sorted_indices.data())
                );

                return sorted_indices;        
            }
            case 2: {
                auto zip_vectors = thrust::make_zip_iterator(
                    maelstrom::device_tptr_cast<T1>(vectors[0].get().data()),
                    maelstrom::device_tptr_cast<T2>(vectors[1].get().data())
                );

                thrust::sort_by_key(
                    exec_policy,
                    zip_vectors,
                    zip_vectors + vectors.front().get().size(),
                    maelstrom::device_tptr_cast<size_t>(sorted_indices.data())
                );

                return sorted_indices; 
            }
            case 3: {
                auto zip_vectors = thrust::make_zip_iterator(
                    maelstrom::device_tptr_cast<T1>(vectors[0].get().data()),
                    maelstrom::device_tptr_cast<T2>(vectors[1].get().data()),
                    maelstrom::device_tptr_cast<T3>(vectors[2].get().data())
                );

                thrust::sort_by_key(
                    exec_policy,
                    zip_vectors,
                    zip_vectors + vectors.front().get().size(),
                    maelstrom::device_tptr_cast<size_t>(sorted_indices.data())
                );

                return sorted_indices; 
            }
        }

        throw std::runtime_error("Too many vectors provided to t_sort_uint_index (max is 3)");
    }

    template <typename E, typename T1, typename T2>
    maelstrom::vector sort_uint_index_dispatch_three(E exec_policy, std::vector<std::reference_wrapper<maelstrom::vector>> vectors) {
        if(vectors.size() < 3) {
            return sort_uint_index_dispatch_three<E, T1, uint64_t>(exec_policy, vectors);
        }

        switch(vectors[2].get().get_dtype().prim_type) {
            case UINT64:
                return t_sort_uint_index<E, T1, T2, uint64_t>(exec_policy, vectors);
            case UINT32:
                return t_sort_uint_index<E, T1, T2, uint32_t>(exec_policy, vectors);
        }

        throw std::runtime_error("Invalid dtype for sort uint index");
    }

    template <typename E, typename T1>
    maelstrom::vector sort_uint_index_dispatch_two(E exec_policy, std::vector<std::reference_wrapper<maelstrom::vector>> vectors) {
        if(vectors.size() < 2) {
            return sort_uint_index_dispatch_three<E, T1, uint64_t>(exec_policy, vectors);
        }

        switch(vectors[1].get().get_dtype().prim_type) {
            case UINT64:
                return sort_uint_index_dispatch_three<E, T1, uint64_t>(exec_policy, vectors);
            case UINT32:
                return sort_uint_index_dispatch_three<E, T1, uint32_t>(exec_policy, vectors);
        }

        throw std::runtime_error("Invalid dtype for sort uint index");
    }

    template <typename E>
    maelstrom::vector sort_uint_index_dispatch_one(E exec_policy, std::vector<std::reference_wrapper<maelstrom::vector>> vectors) {
        switch(vectors.front().get().get_dtype().prim_type) {
            case UINT64:
                return sort_uint_index_dispatch_two<E, uint64_t>(exec_policy, vectors);
            case UINT32:
                return sort_uint_index_dispatch_two<E, uint32_t>(exec_policy, vectors);
        }

        throw std::runtime_error("Invalid dtype for sort uint index");
    }

    maelstrom::vector sort_uint_index_dispatch_exec_policy(std::vector<std::reference_wrapper<maelstrom::vector>> vectors) {
        std::any exec_policy = maelstrom::get_execution_policy(vectors.front().get()).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return sort_uint_index_dispatch_one(
                std::any_cast<device_exec_t>(exec_policy),
                std::move(vectors)
            );
        } else if(typeid(host_exec_t) == t) {
            return sort_uint_index_dispatch_one(
                std::any_cast<host_exec_t>(exec_policy),
                std::move(vectors)
            );
        }

        throw std::runtime_error("Invalid execution policy for sort uint index");
    }

}