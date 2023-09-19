#include "maelstrom/algorithms/sort.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    extern maelstrom::vector sort_uint_index_dispatch_exec_policy(std::vector<std::reference_wrapper<maelstrom::vector>> vectors);

    template<typename E, typename T>
    maelstrom::vector t_sort(E exec_policy, std::vector<std::reference_wrapper<maelstrom::vector>> vectors) {
        auto sorted_indices = maelstrom::arange(vectors.front().get().get_mem_type(), vectors.front().get().size());

        switch(vectors.size()) {
            case 0:
                throw std::runtime_error("no vectors provided to sort");
            case 1: {
                thrust::sort_by_key(
                    exec_policy,
                    maelstrom::device_tptr_cast<T>(vectors.front().get().data()),
                    maelstrom::device_tptr_cast<T>(vectors.front().get().data()) + vectors.front().get().size(),
                    maelstrom::device_tptr_cast<size_t>(sorted_indices.data())
                );

                return sorted_indices;        
            }
            case 2: {
                auto zip_vectors = thrust::make_zip_iterator(
                    maelstrom::device_tptr_cast<T>(vectors[0].get().data()),
                    maelstrom::device_tptr_cast<T>(vectors[1].get().data())
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
                    maelstrom::device_tptr_cast<T>(vectors[0].get().data()),
                    maelstrom::device_tptr_cast<T>(vectors[1].get().data()),
                    maelstrom::device_tptr_cast<T>(vectors[2].get().data())
                );

                thrust::sort_by_key(
                    exec_policy,
                    zip_vectors,
                    zip_vectors + vectors.front().get().size(),
                    maelstrom::device_tptr_cast<size_t>(sorted_indices.data())
                );

                return sorted_indices; 
            }
            case 4: {
                auto zip_vectors = thrust::make_zip_iterator(
                    maelstrom::device_tptr_cast<T>(vectors[0].get().data()),
                    maelstrom::device_tptr_cast<T>(vectors[1].get().data()),
                    maelstrom::device_tptr_cast<T>(vectors[2].get().data()),
                    maelstrom::device_tptr_cast<T>(vectors[3].get().data())
                );

                thrust::sort_by_key(
                    exec_policy,
                    zip_vectors,
                    zip_vectors + vectors.front().get().size(),
                    maelstrom::device_tptr_cast<size_t>(sorted_indices.data())
                );

                return sorted_indices; 
            }
            case 5: {
                auto zip_vectors = thrust::make_zip_iterator(
                    maelstrom::device_tptr_cast<T>(vectors[0].get().data()),
                    maelstrom::device_tptr_cast<T>(vectors[1].get().data()),
                    maelstrom::device_tptr_cast<T>(vectors[2].get().data()),
                    maelstrom::device_tptr_cast<T>(vectors[3].get().data()),
                    maelstrom::device_tptr_cast<T>(vectors[4].get().data())
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

        throw std::runtime_error("too many vectors to sort, max is 5");
    }

    template <typename E>
    maelstrom::vector sort_dispatch_val(E exec_policy, std::vector<std::reference_wrapper<maelstrom::vector>> vectors) {
        switch(vectors.front().get().get_dtype().prim_type) {
            case UINT64:
                return t_sort<E, uint64_t>(exec_policy, std::move(vectors));
            case UINT32:
                return t_sort<E, uint32_t>(exec_policy, std::move(vectors));
            case UINT8:
                return t_sort<E, uint8_t>(exec_policy, std::move(vectors));
            case INT64:
                return t_sort<E, int64_t>(exec_policy, std::move(vectors));
            case INT32:
                return t_sort<E, int32_t>(exec_policy, std::move(vectors));
            case INT8:
                return t_sort<E, int8_t>(exec_policy, std::move(vectors));
            case FLOAT64:
                return t_sort<E, double>(exec_policy, std::move(vectors));
            case FLOAT32:
                return t_sort<E, float>(exec_policy, std::move(vectors));
        }

        throw std::runtime_error("Invalid dtype provided to sort()");
    }

    maelstrom::vector sort_dispatch_exec_policy(std::vector<std::reference_wrapper<maelstrom::vector>> vectors) {
        std::any exec_policy = maelstrom::get_execution_policy(vectors.front().get()).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return sort_dispatch_val(
                std::any_cast<device_exec_t>(exec_policy),
                std::move(vectors)
            );
        } else if(typeid(host_exec_t) == t) {
            return sort_dispatch_val(
                std::any_cast<host_exec_t>(exec_policy),
                std::move(vectors)
            );
        }

        throw std::runtime_error("Invalid execution policy for sort");
    }

    maelstrom::vector sort(std::vector<std::reference_wrapper<maelstrom::vector>> vectors) {
        if(vectors.size() > 1) {
            auto dtype = vectors.front().get().get_dtype();
            auto sz = vectors.front().get().size();
            bool dtype_mismatch = false;
            bool uint64_uint32_prim_only = true;
            for(auto& vec : vectors) {
                auto current_dtype = vec.get().get_dtype();
                if(current_dtype != dtype) dtype_mismatch = true;
                if(current_dtype.prim_type != UINT64 && current_dtype.prim_type != UINT32 ) uint64_uint32_prim_only = false;
                if(vec.get().size() != sz) throw std::runtime_error("Sizes of vectors to be sorted must match!");
                // TODO check memtype compatibility
            }

            if(dtype_mismatch) {
                if(uint64_uint32_prim_only) {
                    if(vectors.size() <= 3) return sort_uint_index_dispatch_exec_policy(vectors);
                    else throw std::runtime_error("Sort index only supported with 3 or fewer vectors");
                }
                throw std::runtime_error("Types of vectors to be sorted must match!");
            }
        }
        return sort_dispatch_exec_policy(std::move(vectors));
    }

}