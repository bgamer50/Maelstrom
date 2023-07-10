#include "maelstrom/algorithms/sort.h"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    template<typename E, typename T>
    maelstrom::vector t_sort(E exec_policy, std::vector<std::reference_wrapper<maelstrom::vector>> vectors) {
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
        boost::any exec_policy = maelstrom::get_execution_policy(vectors.front().get()).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return sort_dispatch_val(
                boost::any_cast<device_exec_t>(exec_policy),
                std::move(vectors)
            );
        } else if(typeid(host_exec_t) == t) {
            return sort_dispatch_val(
                boost::any_cast<host_exec_t>(exec_policy),
                std::move(vectors)
            );
        }

        throw std::runtime_error("Invalid execution policy for sort");
    }

    maelstrom::vector sort(std::vector<std::reference_wrapper<maelstrom::vector>> vectors) {
        if(vectors.size() > 1) {
            auto dtype = vectors.front().get().get_dtype();
            auto mem_type = vectors.front().get().get_mem_type();
            auto sz = vectors.front().get().size();
            for(auto& vec : vectors) {
                if(vec.get().get_dtype() != dtype) throw std::runtime_error("Data types in vectors to be sorted must match!");
                if(vec.get().get_mem_type() != mem_type) throw std::runtime_error("Memory types in vectors to be sorted must match!");
                if(vec.get().size() != sz) throw std::runtime_error("Sizes of vectors to be sorted must match!");
            }
        }
        return sort_dispatch_exec_policy(std::move(vectors));
    }

}