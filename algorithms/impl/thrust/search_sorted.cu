#include "maelstrom/algorithms/search_sorted.h"
#include "maelstrom/thrust_utils/execution.cuh"

namespace maelstrom {

    extern maelstrom::vector search_sorted_host_dispatch_val(maelstrom::vector& sorted_array, maelstrom::vector& values_to_find);
    
    template<typename E> 
    extern maelstrom::vector search_sorted_device_dispatch_val(E exec_policy, maelstrom::vector& sorted_array, maelstrom::vector& values_to_find);

    maelstrom::vector search_sorted_dispatch_exec_policy(maelstrom::vector& sorted_array, maelstrom::vector& values_to_find) {
        std::any exec_policy = maelstrom::get_execution_policy(sorted_array).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return search_sorted_device_dispatch_val(
                std::any_cast<device_exec_t>(exec_policy),
                sorted_array,
                values_to_find
            );
        } else if(typeid(host_exec_t) == t) {
            return search_sorted_host_dispatch_val(
                sorted_array,
                values_to_find
            );
        }

        throw std::runtime_error("Invalid execution policy for search sorted");
    }

    maelstrom::vector search_sorted(maelstrom::vector& sorted_array, maelstrom::vector& values_to_find) {
        if(sorted_array.get_dtype() != values_to_find.get_dtype()) throw std::runtime_error("Data types must match to search!");
        if(sorted_array.get_mem_type() != values_to_find.get_mem_type()) throw std::runtime_error("Memory types must match to search!");

        return search_sorted_dispatch_exec_policy(sorted_array, values_to_find);
    }

}