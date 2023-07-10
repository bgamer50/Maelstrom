#include "maelstrom/algorithms/sparse/search_sorted_sparse.h"
#include "maelstrom/thrust_utils/execution.cuh"

namespace maelstrom {

    namespace sparse {

        extern maelstrom::vector search_sorted_sparse_host_dispatch_ix(maelstrom::vector& row, maelstrom::vector& col, maelstrom::vector& ix_r, maelstrom::vector& ix_c, boost::any index_not_found);
        
        template <typename E>
        extern maelstrom::vector search_sorted_sparse_device_dispatch_ix(E exec_policy, maelstrom::vector& row, maelstrom::vector& col, maelstrom::vector& ix_r, maelstrom::vector& ix_c, boost::any index_not_found);

        maelstrom::vector search_sorted_sparse_dispatch_exec_policy(maelstrom::vector& row, maelstrom::vector& col, maelstrom::vector& ix_r, maelstrom::vector& ix_c, boost::any index_not_found) {
            boost::any exec_policy = maelstrom::get_execution_policy(row).get();
            const std::type_info& t = exec_policy.type();
            
            if(typeid(device_exec_t) == t) {
                return search_sorted_sparse_device_dispatch_ix(
                    boost::any_cast<device_exec_t>(exec_policy),
                    row,
                    col,
                    ix_r,
                    ix_c,
                    index_not_found
                );
            } else if(typeid(host_exec_t) == t) {
                return search_sorted_sparse_host_dispatch_ix(
                    row,
                    col,
                    ix_r,
                    ix_c,
                    index_not_found
                );
            }

            throw std::runtime_error("Invalid execution policy for search sorted sparse");
        }

        maelstrom::vector search_sorted_sparse(maelstrom::vector& row, maelstrom::vector& col, maelstrom::vector& ix_r, maelstrom::vector& ix_c, boost::any index_not_found) {
            if(row.get_dtype() != col.get_dtype()) throw std::runtime_error("Row and col data type must match");
            if(ix_r.get_dtype() != row.get_dtype()) throw std::runtime_error("row index data type must match row data type");
            if(ix_r.get_dtype() != ix_c.get_dtype()) throw std::runtime_error("column index data type must match row index data type");

            if(row.get_mem_type() != col.get_mem_type()) throw std::runtime_error("Row and col memory type must match");
            if(ix_r.get_mem_type() != row.get_mem_type()) throw std::runtime_error("row index memory type must match row memory type");
            if(ix_r.get_mem_type() != ix_c.get_mem_type()) throw std::runtime_error("column index memory type must match row index memory type");

            if(ix_r.size() != ix_c.size()) throw std::runtime_error("row index and column index must be of same size");

            return search_sorted_sparse_dispatch_exec_policy(
                row,
                col,
                ix_r,
                ix_c,
                index_not_found
            );
        }

    }

}