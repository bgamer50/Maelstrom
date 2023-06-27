#include "maelstrom/containers/sparse_matrix.h"
#include "maelstrom/algorithms/search_sorted.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/increment.h"

namespace maelstrom {

    maelstrom::vector basic_sparse_matrix::get_rows_1d(maelstrom::vector& ix_1d) {
        if(this->format == COO || this->format == CSC) {
            return maelstrom::select(this->row, ix_1d);
        }

        // CSR
        auto row = maelstrom::search_sorted(this->row, ix_1d);
        maelstrom::increment(row, true); // decrement by 1
        return row;
    }

    maelstrom::vector basic_sparse_matrix::get_cols_1d(maelstrom::vector& ix_1d) {
        if(this->format == COO || this->format == CSR) {
            return maelstrom::select(this->col, ix_1d);
        }

        // CSC
        auto col = maelstrom::search_sorted(this->col, ix_1d);
        maelstrom::increment(col, true); // decrement by 1
        return col;
    }

    maelstrom::vector basic_sparse_matrix::get_values_1d(maelstrom::vector& ix_1d) {
        return maelstrom::select(this->val, ix_1d);
    }

    std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector> maelstrom::basic_sparse_matrix::get_entries_1d(maelstrom::vector& ix_1d) {
        // TODO validity checking of edge ids

        return std::make_tuple(
            std::move(this->get_rows_1d(ix_1d)),
            std::move(this->get_cols_1d(ix_1d)),
            std::move(this->get_values_1d(ix_1d))
        );
    }

    virtual maelstrom::vector get_values_2d(maelstrom::vector& ix_r, maelstrom::vector& ix_c) {
        if(this->format == CSR) {
            // call kernel
        } else if(this->format == CSC) {
            // call same kernel with arguments reversed
        } else {
            // sort the coo if it isn't already
        }
    }

    virtual void set(maelstrom::vector& rows, maelstrom::vector& cols, std::optional<maelstrom::vector&> vals=std::nullopt);

    virtual void sort();

}