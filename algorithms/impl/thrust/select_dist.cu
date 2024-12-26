#include "maelstrom/containers/vector.h"

#include "maelstrom/algorithms/prefix_sum.h"
#include "maelstrom/algorithms/search_sorted.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/increment.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/sort.h"

#include "maelstrom/algorithms/dist/shuffle.h"

#include "maelstrom/dist_utils/dist_partition.cuh"

namespace maelstrom {

    maelstrom::vector select_dispatch_dist(maelstrom::vector& vec, maelstrom::vector& idx) {
        size_t rank = maelstrom::get_rank();

        auto partition_sizes = maelstrom::get_current_partition_sizes(vec);
        maelstrom::prefix_sum(partition_sizes);

        auto rix = maelstrom::search_sorted(partition_sizes, idx);
        auto idx_dist_copy = maelstrom::to_dist_vector(idx);

        std::cout << rix.size() << " " << rix.local_size() << std::endl;
        
        maelstrom::shuffle(idx_dist_copy, rix); // TODO have to reindex so the order in idx is preserved

        auto arix = maelstrom::vector(
            vec.get_mem_type(),
            maelstrom::uint64,
            rix.size(),
            rix.local_size()
        );
        maelstrom::set(arix, (size_t)rank);
        maelstrom::shuffle(arix, rix);

        auto local_index = maelstrom::arange(vec.get_mem_type(), rix.local_size());
        maelstrom::shuffle(local_index, rix);

        rix.clear();

        size_t offset = (rank == 0) ? 0 : std::any_cast<size_t>(partition_sizes.get(rank - 1));
        maelstrom::increment(idx_dist_copy, offset, maelstrom::DECREMENT);
        
        auto vec_local_view = maelstrom::local_view_of(vec);
        auto idx_copy_local_view = maelstrom::local_view_of(idx_dist_copy);
        auto vals = maelstrom::select(vec_local_view, idx_copy_local_view);

        idx_dist_copy.clear();
        maelstrom::shuffle(vals, arix);
        maelstrom::shuffle(local_index, arix);
        arix.clear();

        auto local_index_local_view = maelstrom::local_view_of(local_index);
        auto six = maelstrom::sort(local_index_local_view);
        local_index.clear();

        auto vals_local_view = maelstrom::local_view_of(vals);
        auto final_vals = maelstrom::select(vals_local_view, six);
        vals.clear();

        return maelstrom::to_dist_vector(std::move(final_vals));
    }

}