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

    maelstrom::vector basic_sparse_matrix::get_relations_1d(maelstrom::vector& ix_1d) {
        return maelstrom::select(this->rel, ix_1d);
    }

    std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> maelstrom::basic_sparse_matrix::get_entries_1d(maelstrom::vector& ix_1d) {
        // TODO validity checking of edge ids

        return std::make_tuple(
            std::move(this->get_rows_1d(ix_1d)),
            std::move(this->get_cols_1d(ix_1d)),
            std::move(this->get_values_1d(ix_1d)),
            std::move(this->get_relations_1d(ix_1d))
        );
    }

    maelstrom::vector basic_sparse_matrix::get_1d_index_from_2d_index(maelstrom::vector& ix_r, maelstrom::vector& ix_c) {
        if(!this->sorted) this->sort(); // sort the col ptr

        if(this->format == CSR) {
            return maelstrom::sparse::search_sorted_sparse(this->row, this->col, ix_r, ix_c);
        } else if(this->format == CSC) {
            return maelstrom::sparse::search_sorted_sparse(this->col, this->row, ix_c, ix_r);
        }

        // COO
        throw std::runtime_error("2d-indexing a COO matrix is currently unsupported");        
    }

    std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> basic_sparse_matrix::query_adjacency(maelstrom::vector& ix, maelstrom::vector& rel_types, bool return_inner=true, bool return_values=false, bool return_relations=false) {
        // sorting is not required

        if(this->format == CSR) {
            return maelstrom::sparse::query_adjacency(
                this->row,
                this->col,
                this->val,
                this->rel,
                ix,
                rel_types,
                return_inner,
                return_values,
                return_relations
            );
        } else if(this->format == CSC) {
            return maelstrom::sparse::query_adjacency(
                this->col,
                this->row,
                this->val,
                this->rel,
                ix,
                rel_types,
                return_inner,
                return_values,
                return_relations
            );
        }

        throw std::runtime_error("adj-querying a COO matrix is unsupported");
    }

    virtual void set(maelstrom::vector& rows, maelstrom::vector& cols, std::optional<maelstrom::vector&> vals=std::nullopt);

    virtual void sort();

}