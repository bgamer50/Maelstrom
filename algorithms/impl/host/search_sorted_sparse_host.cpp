#include "maelstrom/containers/vector.h"
#include "maelstrom/util/any_utils.cuh"
#include <limits>

namespace maelstrom {
    namespace sparse {

        template <typename T>
        void h_search_sorted_sparse(T* row, T* col, T* ix_r, T* ix_c, T* output_ix, size_t ix_size, T default_index) {
            // TODO parallelize
            for(size_t k = 0; k < ix_size; ++k) {
                const T row_idx = ix_r[k];
                const T row_start = row[row_idx];
                const T row_end = row[row_idx+1];

                // Find the first index of the column
                const T value = ix_c[k];

                T left = row_start;
                T right = row_end;
                
                while(left < right) {
                    const T i = (left + right) / 2;
                    const T lower_value = col[i];

                    if(lower_value < value) left = i + 1;
                    else right = i;
                }

                output_ix[k] = (col[left] == value) ? left : default_index;
            }
        }

        template <typename T>
        maelstrom::vector launch_search_sorted_sparse_host(maelstrom::vector& row, maelstrom::vector& col, maelstrom::vector& ix_r, maelstrom::vector& ix_c, boost::any index_not_found) {
            maelstrom::vector output_ix(row.get_mem_type(), row.get_dtype(), ix_r.size());

            h_search_sorted_sparse(
                static_cast<T*>(row.data()),
                static_cast<T*>(col.data()),
                static_cast<T*>(ix_r.data()),
                static_cast<T*>(ix_c.data()),
                static_cast<T*>(output_ix.data()),
                ix_r.size(),
                index_not_found.empty() ? std::numeric_limits<T>::max() : boost::any_cast<T>(maelstrom::safe_any_cast(index_not_found, ix_r.get_dtype()))
            );

            return output_ix;
        }

        maelstrom::vector search_sorted_sparse_host_dispatch_ix(maelstrom::vector& row, maelstrom::vector& col, maelstrom::vector& ix_r, maelstrom::vector& ix_c, boost::any index_not_found) {
            switch(row.get_dtype().prim_type) {
                case UINT64:
                    return launch_search_sorted_sparse_host<uint64_t>(row, col, ix_r, ix_c, index_not_found);
                case UINT32:
                    return launch_search_sorted_sparse_host<uint32_t>(row, col, ix_r, ix_c, index_not_found);
                case UINT8:
                    return launch_search_sorted_sparse_host<uint8_t>(row, col, ix_r, ix_c, index_not_found);
                case INT64:
                    return launch_search_sorted_sparse_host<int64_t>(row, col, ix_r, ix_c, index_not_found);
                case INT32:
                    return launch_search_sorted_sparse_host<int32_t>(row, col, ix_r, ix_c, index_not_found);
            }

            throw std::runtime_error("unsupported dtype for search sorted sparse");
        }

    }
}
