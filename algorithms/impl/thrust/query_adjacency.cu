#include "maelstrom/algorithms/sparse/query_adjacency.h"
#include "maelstrom/thrust_utils/execution.cuh"

namespace maelstrom {
    namespace sparse {

        extern std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency_host_dispatch_ix(maelstrom::vector& row,
                                                                                                                                       maelstrom::vector& col,
                                                                                                                                       maelstrom::vector& val,
                                                                                                                                       maelstrom::vector& rel,
                                                                                                                                       maelstrom::vector& ix,
                                                                                                                                       maelstrom::vector& rel_types,
                                                                                                                                       bool return_inner,
                                                                                                                                       bool return_values,
                                                                                                                                       bool return_relations,
                                                                                                                                       bool return_1d_index_as_values);
    
        template<typename E> 
        extern std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency_device_dispatch_ix(E execution_policy,
                                                                                                                                         maelstrom::vector& row,
                                                                                                                                         maelstrom::vector& col,
                                                                                                                                         maelstrom::vector& val,
                                                                                                                                         maelstrom::vector& rel,
                                                                                                                                         maelstrom::vector& ix,
                                                                                                                                         maelstrom::vector& rel_types,
                                                                                                                                         bool return_inner,
                                                                                                                                         bool return_values,
                                                                                                                                         bool return_relations,
                                                                                                                                         bool return_1d_index_as_values);

        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency_dispatch_exec_policy(maelstrom::vector& row,
                                                                                                                                    maelstrom::vector& col,
                                                                                                                                    maelstrom::vector& val,
                                                                                                                                    maelstrom::vector& rel,
                                                                                                                                    maelstrom::vector& ix,
                                                                                                                                    maelstrom::vector& rel_types,
                                                                                                                                    bool return_inner,
                                                                                                                                    bool return_values,
                                                                                                                                    bool return_relations,
                                                                                                                                    bool return_1d_index_as_values)
        {
            std::any exec_policy = maelstrom::get_execution_policy(row).get();
            const std::type_info& t = exec_policy.type();
            
            if(typeid(device_exec_t) == t) {
                return query_adjacency_device_dispatch_ix(
                    std::any_cast<device_exec_t>(exec_policy),
                    row,
                    col,
                    val,
                    rel,
                    ix,
                    rel_types,
                    return_inner,
                    return_values,
                    return_relations,
                    return_1d_index_as_values
                );
            } else if(typeid(host_exec_t) == t) {
                return query_adjacency_host_dispatch_ix(
                    row,
                    col,
                    val,
                    rel,
                    ix,
                    rel_types,
                    return_inner,
                    return_values,
                    return_relations,
                    return_1d_index_as_values
                );
            }

            throw std::runtime_error("Invalid execution policy");
        }

        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency(maelstrom::vector& row,
                                                                                                               maelstrom::vector& col,
                                                                                                               maelstrom::vector& val,
                                                                                                               maelstrom::vector& rel,
                                                                                                               maelstrom::vector& ix,
                                                                                                               maelstrom::vector& rel_types,
                                                                                                               bool return_inner,
                                                                                                               bool return_values,
                                                                                                               bool return_relations,
                                                                                                               bool return_1d_index_as_values)
        {
            if(row.get_dtype() != col.get_dtype()) throw std::runtime_error("dtype of row and col must match");
            if(ix.get_dtype() != col.get_dtype()) throw std::runtime_error("index dtype must match row/col dtype");

            if(row.get_mem_type() != col.get_mem_type()) throw std::runtime_error("row and col must have the same memory type");
            if(ix.get_mem_type() != col.get_mem_type()) throw std::runtime_error("index must have the same memory type as row/col");

            if(!rel_types.empty() && rel_types.get_dtype() != rel.get_dtype()) throw std::runtime_error("query relation dtype must match relation dtype");
            
            return query_adjacency_dispatch_exec_policy(
                row,
                col,
                val,
                rel,
                ix,
                rel_types,
                return_inner,
                return_values,
                return_relations,
                return_1d_index_as_values
            );

        }

    }
}