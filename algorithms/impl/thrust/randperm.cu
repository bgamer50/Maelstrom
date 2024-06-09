#include "maelstrom/algorithms/randperm.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

#include <thrust/shuffle.h>

namespace maelstrom {

    maelstrom::vector randperm(maelstrom::storage mem_type, size_t array_size, size_t num_to_select, std::optional<size_t> seed) {
        if(num_to_select > array_size) throw std::invalid_argument("num_to_select must be <= array_size");
        if(num_to_select == 0 || array_size == 0) throw std::invalid_argument("num_to_select and array_size must be > 0");

        auto perm = maelstrom::arange(mem_type, array_size);

        thrust::default_random_engine rand;
        if(seed) rand = thrust::default_random_engine(seed.value());
        
        std::any exec_policy = maelstrom::get_execution_policy(perm).get();
        const std::type_info& t = exec_policy.type();

        if(typeid(device_exec_t) == t) {
            thrust::shuffle(
                std::any_cast<device_exec_t>(exec_policy),
                maelstrom::device_tptr_cast<size_t>(perm.data()),
                maelstrom::device_tptr_cast<size_t>(perm.data()) + perm.size(),
                rand
            );
        } else if(typeid(host_exec_t) == t) {
            thrust::shuffle(
                std::any_cast<host_exec_t>(exec_policy),
                maelstrom::device_tptr_cast<size_t>(perm.data()),
                maelstrom::device_tptr_cast<size_t>(perm.data()) + perm.size(),
                rand
            );
        } else {
            throw std::runtime_error("Invalid execution policy for randperm");
        }

        perm.resize(num_to_select);
        perm.shrink_to_fit();
        return perm;
    }

}