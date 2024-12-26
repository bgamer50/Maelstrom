#include "maelstrom/algorithms/dist/bucket_sort.h"
#include "maelstrom/algorithms/dist/shuffle.h"
#include "maelstrom/algorithms/dist/rebalance.h"

#include "maelstrom/algorithms/sort.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/search_sorted.h"
#include "maelstrom/algorithms/prefix_sum.h"
#include "maelstrom/algorithms/increment.h"
#include "maelstrom/algorithms/arange.h"

#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/dist_utils/dist_partition.cuh"
#include "maelstrom/dist_utils/nccl_utils.cuh"

namespace maelstrom {

    maelstrom::vector bucket_sort(maelstrom::vector& vec, maelstrom::sort_partition_type_t partition_behavior) {
        if(!is_dist(vec.get_mem_type())) {
            return maelstrom::sort(vec);
        }

        size_t world_size = maelstrom::get_world_size();
        size_t rank = maelstrom::get_rank();
        auto dtype = vec.get_dtype();

        // Sort the local arrays
        auto vec_local_view = maelstrom::local_view_of(vec);
        maelstrom::vector local_ix = maelstrom::sort(vec_local_view);

        maelstrom::vector medians(maelstrom::DIST_DEVICE, dtype, world_size);
        auto medians_local_view = maelstrom::local_view_of(medians);
        
        maelstrom::shuffle_to_rank(medians, 0);

        if(rank == 0) {
            medians_local_view = maelstrom::local_view_of(medians);
            maelstrom::sort(medians_local_view);
        }

        maelstrom::vector buckets(maelstrom::DEVICE, dtype, world_size);
        maelstrom::nccl::ncclCheckErrors(
            ncclBroadcast(
                medians.data(),
                buckets.data(),
                maelstrom::size_of(dtype) * world_size,
                ncclUint8,
                0,
                maelstrom::get_nccl_comms(),
                std::any_cast<cudaStream_t>(vec.get_stream())
            ),
            "bucket sort broadcast medians"
        );
        cudaStreamSynchronize(std::any_cast<cudaStream_t>(vec.get_stream()));

        // want world_size buckets, not world_size+1 buckets        
        buckets.resize(buckets.size() - 1);

        auto rix = maelstrom::search_sorted(
            buckets,
            vec
        );
        auto rix_local_view = maelstrom::local_view_of(rix);

        // need to set the proper index
        auto partition_sizes = maelstrom::get_current_partition_sizes(vec);
        maelstrom::prefix_sum(partition_sizes);

        size_t offset;
        if(rank == 0) {
            offset = 0;
        } else {
            offset = std::any_cast<size_t>(partition_sizes.get(rank - 1));
        }
        maelstrom::increment(local_ix, offset);

        local_ix = maelstrom::to_dist_vector(std::move(local_ix));

        maelstrom::shuffle(vec, rix_local_view);
        maelstrom::shuffle(local_ix, rix_local_view);
        rix.clear();

        vec_local_view = maelstrom::local_view_of(vec); // have to recreate the view since the old view is now invalid
        auto six = maelstrom::sort(vec_local_view);
        maelstrom::select(local_ix, six);

        if(partition_behavior == ORIGINAL) {
            auto current_partition_sizes = maelstrom::get_current_partition_sizes(vec);
            maelstrom::prefix_sum(current_partition_sizes);
            auto start = (rank == 0) ? 0 : std::any_cast<size_t>(current_partition_sizes.get(rank - 1));
            auto end = std::any_cast<size_t>(current_partition_sizes.get(rank));

            auto zix = maelstrom::arange(maelstrom::DEVICE, start, end);

            partition_sizes = partition_sizes.to(maelstrom::DEVICE);
            zix = maelstrom::search_sorted(
                partition_sizes,
                zix
            );

            maelstrom::shuffle(vec, zix);
            maelstrom::shuffle(local_ix, zix);
        } else if(partition_behavior == BALANCED) {
            maelstrom::rebalance(vec);
            maelstrom::rebalance(local_ix);
        }

        return local_ix;
    }

}