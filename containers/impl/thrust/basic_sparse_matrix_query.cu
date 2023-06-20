#include "maelstrom/containers/sparse_matrix.h"
#include "maelstrom/algorithms/search_sorted.h"

namespace maelstrom {

    std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector> maelstrom::basic_sparse_matrix::get_entries_1d(maelstrom::vector& ix_1d) {
        // TODO validity checking of edge ids

        if(this->format == COO) {
            /
            return z
        }

        // we have a problem here - edge ids won't match those in coo
        // need to add an optional edge id permutation or alternatively use values as the edge id
        auto rows = maelstrom::search_sorted(this->ix_1d)
    }

    virtual maelstrom::vector get_entries_2d(maelstrom::vector& ix_r, maelstrom::vector& ix_c);

    virtual void set(maelstrom::vector& rows, maelstrom::vector& cols, std::optional<maelstrom::vector&> vals=std::nullopt);

}