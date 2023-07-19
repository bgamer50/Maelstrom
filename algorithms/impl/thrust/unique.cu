#include "maelstrom/algorithms/unique.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/sort.h"
#include "maelstrom/thrust_utils/thrust_utils.cuh"
#include <any>

namespace maelstrom {

    template<typename E, typename T>
    maelstrom::vector t_unique_from_sorted_vec(E exec_policy, maelstrom::vector& vec) {
        maelstrom::vector unique_indices(
            vec.get_mem_type(),
            uint64,
            vec.size()
        );

        auto end = thrust::unique_by_key_copy(
            exec_policy,
            maelstrom::device_tptr_cast<T>(vec.data()),
            maelstrom::device_tptr_cast<T>(vec.data()) + vec.size(),
            thrust::make_counting_iterator(static_cast<size_t>(0)),         
            thrust::make_discard_iterator(),
            maelstrom::device_tptr_cast<size_t>(unique_indices.data())
        );

        unique_indices.resize(static_cast<size_t>(
            end.second - maelstrom::device_tptr_cast<size_t>(unique_indices.data())
        ));

        if((double)unique_indices.size() / (double)vec.size() < 0.66) {
            unique_indices.shrink_to_fit();
        }

        return unique_indices;
    }

    template<typename E>
    maelstrom::vector unique_from_sorted_vec_dispatch_val(E exec_policy, maelstrom::vector& vec) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return t_unique_from_sorted_vec<E, uint64_t>(exec_policy, vec);
            case UINT32:
                return t_unique_from_sorted_vec<E, uint32_t>(exec_policy, vec);
            case UINT8:
                return t_unique_from_sorted_vec<E, uint8_t>(exec_policy, vec);
            case INT64:
                return t_unique_from_sorted_vec<E, int64_t>(exec_policy, vec);
            case INT32:
                return t_unique_from_sorted_vec<E, int32_t>(exec_policy, vec);
            case INT8:
                return t_unique_from_sorted_vec<E, int8_t>(exec_policy, vec);
            case FLOAT64:
                return t_unique_from_sorted_vec<E, double>(exec_policy, vec);
            case FLOAT32:
                return t_unique_from_sorted_vec<E, float>(exec_policy, vec);
        }

        throw std::runtime_error("Invalid dtype provided to unique_from_sorted_vec()");
    }

    maelstrom::vector unique_from_sorted_vec_dispatch_exec_policy(maelstrom::vector& vec) {
        std::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return unique_from_sorted_vec_dispatch_val(
                std::any_cast<device_exec_t>(exec_policy),
                vec
            );
        } else if(typeid(host_exec_t) == t) {
            return unique_from_sorted_vec_dispatch_val(
                std::any_cast<host_exec_t>(exec_policy),
                vec
            );
        }

        throw std::runtime_error("Invalid execution policy for unique");
    }

    maelstrom::vector unique(maelstrom::vector& vec, bool sorted) {
        if(!sorted) {
            maelstrom::vector vec_copy(vec);
            maelstrom::vector sorted_ix = maelstrom::sort(vec_copy);

            maelstrom::vector unique_ix = unique_from_sorted_vec_dispatch_exec_policy(vec_copy);
            vec_copy.clear();
            return maelstrom::select(sorted_ix, unique_ix);
        }
        
        return unique_from_sorted_vec_dispatch_exec_policy(vec);
    }

}