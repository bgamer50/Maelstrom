#include "maelstrom/algorithms/remove_if.h"
#include "maelstrom/thrust_utils/thrust_utils.cuh"
#include "maelstrom/thrust_utils/execution.cuh"

#include <boost/any.hpp>
#include <sstream>

namespace maelstrom {
    template <typename E, typename T, typename U>
    void t_remove_if(E thrust_exec_policy, maelstrom::vector& array, maelstrom::vector& stencil) {

        auto new_last = thrust::remove_if(
            thrust_exec_policy,
            device_tptr_cast<T>(array.data()),
            device_tptr_cast<T>(array.data()) + array.size(),
            device_tptr_cast<U>(stencil.data()),
            thrust::identity<U>()
        );

        size_t new_size = new_last - device_tptr_cast<T>(array.data());
        array.resize(new_size);
    
    }

    template<typename E, typename T>
    void remove_if_dispatch_inner(E thrust_exec_policy, maelstrom::vector& array, maelstrom::vector& stencil) {
        switch(stencil.get_dtype().prim_type) {
            case maelstrom::primitive_t::UINT8:
                return t_remove_if<E, T, uint8_t>(
                    thrust_exec_policy,
                    array, 
                    stencil
                );
            case maelstrom::primitive_t::UINT32:
                return t_remove_if<E, T, uint32_t>(
                    thrust_exec_policy,
                    array, 
                    stencil
            );
            case maelstrom::primitive_t::UINT64:
                return t_remove_if<E, T, uint64_t>(
                    thrust_exec_policy,
                    array, 
                    stencil
                );
            case maelstrom::primitive_t::INT8:
                return t_remove_if<E, T, int8_t>(
                    thrust_exec_policy,
                    array, 
                    stencil
                );
            case maelstrom::primitive_t::INT32:
                return t_remove_if<E, T, int32_t>(
                    thrust_exec_policy,
                    array, 
                    stencil
                );
            case maelstrom::primitive_t::INT64:
                return t_remove_if<E, T, int64_t>(
                    thrust_exec_policy,
                    array, 
                    stencil
                );
        }

        throw std::runtime_error("Invalid dtype provided to remove_if");
    }

    template<typename E>
    void remove_if_dispatch_outer(E thrust_exec_policy, maelstrom::vector& array, maelstrom::vector& stencil) {
        switch(array.get_dtype().prim_type) {
            case maelstrom::primitive_t::UINT8:
                return remove_if_dispatch_inner<E, uint8_t>(
                    thrust_exec_policy,
                    array,
                    stencil
                );
            case maelstrom::primitive_t::UINT32:
                return remove_if_dispatch_inner<E, uint32_t>(
                    thrust_exec_policy,
                    array,
                    stencil
            );
            case maelstrom::primitive_t::UINT64:
                return remove_if_dispatch_inner<E, uint64_t>(
                    thrust_exec_policy,
                    array,
                    stencil
                );
            case maelstrom::primitive_t::INT8:
                return remove_if_dispatch_inner<E, int8_t>(
                    thrust_exec_policy,
                    array,
                    stencil
                );
            case maelstrom::primitive_t::INT32:
                return remove_if_dispatch_inner<E, int32_t>(
                    thrust_exec_policy,
                    array,
                    stencil
                );
            case maelstrom::primitive_t::INT64:
                return remove_if_dispatch_inner<E, int64_t>(
                    thrust_exec_policy,
                    array,
                    stencil
                );
            case maelstrom::primitive_t::FLOAT32:
                return remove_if_dispatch_inner<E, float>(
                    thrust_exec_policy,
                    array,
                    stencil
                );
            case maelstrom::primitive_t::FLOAT64:
                return remove_if_dispatch_inner<E, double>(
                    thrust_exec_policy,
                    array,
                    stencil
                );
        }

        throw std::runtime_error("Invalid dtype provided to remove_if");
    }

    void remove_if(maelstrom::vector& array, maelstrom::vector& stencil) {
        // Error checking
        if(array.get_mem_type() != stencil.get_mem_type()) {
            std::stringstream sx;
            sx << "Memory type of array (" << array.get_mem_type() << ")";
            sx << " does not match memory type of stencil (" << stencil.get_mem_type() << ")";
            throw std::runtime_error(sx.str());
        }

        boost::any exec_policy = maelstrom::get_execution_policy(array).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return remove_if_dispatch_outer(
                boost::any_cast<device_exec_t>(exec_policy),
                array,
                stencil
            );
        } else if(typeid(host_exec_t) == t) {
            return remove_if_dispatch_outer(
                boost::any_cast<host_exec_t>(exec_policy),
                array,
                stencil
            );
        }

        throw std::runtime_error("Invalid execution policy");
    }

}