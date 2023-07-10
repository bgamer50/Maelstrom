#include "maelstrom/containers/vector.h"

namespace maelstrom {
    namespace sparse {
        template<typename T>
        void h_csr_to_coo(T* csr, T* coo, size_t n_rows) {
            // TODO parallelize this

            for(T k = 0; k < n_rows; ++k) {
                const T start = csr[k];
                const T d = csr[k+1] - csr[k];
                for(T i = 0; i < d; ++i) {
                    coo[start+i] = k;
                }
            }
        }

        template <typename T>
        maelstrom::vector launch_csr_to_coo_host(maelstrom::vector& ptr, size_t nnz) {
            maelstrom::vector coo(
                ptr.get_mem_type(),
                ptr.get_dtype(),
                nnz
            );

            h_csr_to_coo(
                static_cast<T*>(ptr.data()),
                static_cast<T*>(coo.data()),
                ptr.size() - 1
            );

            return coo;
        }

        maelstrom::vector csr_to_coo_host_dispatch_val(maelstrom::vector& ptr, size_t nnz) {
            switch(ptr.get_dtype().prim_type) {
                case UINT64:
                    return launch_csr_to_coo_host<uint64_t>(ptr, nnz);
                case UINT32:
                    return launch_csr_to_coo_host<uint32_t>(ptr, nnz);
                case UINT8:
                    return launch_csr_to_coo_host<uint8_t>(ptr, nnz);
                case INT64:
                    return launch_csr_to_coo_host<int64_t>(ptr, nnz);
                case INT32:
                    return launch_csr_to_coo_host<int32_t>(ptr, nnz);
            }

            throw std::runtime_error("Invalid dtype for csr_to_coo");
        }
    }
}