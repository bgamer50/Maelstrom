#include "maelstrom/algorithms/sparse/csr_to_coo.h"
#include "maelstrom/thrust_utils/execution.cuh"

namespace maelstrom {
    namespace sparse {

        template <typename E>
        extern maelstrom::vector csr_to_coo_device_dispatch_val(E exec_policy, maelstrom::vector& ptr, size_t nnz);
    
        extern maelstrom::vector csr_to_coo_host_dispatch_val(maelstrom::vector& ptr, size_t nnz);

        maelstrom::vector csr_to_coo_dispatch_exec_policy(maelstrom::vector& ptr, size_t nnz) {
            boost::any exec_policy = maelstrom::get_execution_policy(ptr).get();
            const std::type_info& t = exec_policy.type();
            
            if(typeid(device_exec_t) == t) {
                return csr_to_coo_device_dispatch_val(
                    boost::any_cast<device_exec_t>(exec_policy),
                    ptr,
                    nnz
                );
            } else if(typeid(host_exec_t) == t) {
                return csr_to_coo_host_dispatch_val(
                    ptr,
                    nnz
                );
            }

            throw std::runtime_error("Invalid execution policy for csr_to_coo");
        }

        maelstrom::vector csr_to_coo(maelstrom::vector& ptr, size_t nnz) {
            return csr_to_coo_dispatch_exec_policy(ptr, nnz);
        }

    }
}