#include "maelstrom/containers/sparse_matrix.h"

#include "maelstrom/algorithms/search_sorted.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/increment.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/filter.h"
#include "maelstrom/algorithms/sort.h"

#include "maelstrom/algorithms/sparse/query_adjacency.h"
#include "maelstrom/algorithms/sparse/search_sorted_sparse.h"

#include "maelstrom/util/any_utils.cuh"

namespace maelstrom {

    maelstrom::vector basic_sparse_matrix::get_rows_1d(maelstrom::vector& ix_1d) {
        if(this->format == COO || this->format == CSC) {
            return maelstrom::select(this->row, ix_1d);
        }

        // CSR
        auto row = maelstrom::search_sorted(this->row, ix_1d);
        if(row.get_dtype() != this->row.get_dtype()) row = row.astype(this->row.get_dtype());
        maelstrom::increment(row, DECREMENT); // decrement by 1
        return row;
    }

    maelstrom::vector basic_sparse_matrix::get_cols_1d(maelstrom::vector& ix_1d) {
        if(this->format == COO || this->format == CSR) {
            return maelstrom::select(this->col, ix_1d);
        }

        // CSC
        auto col = maelstrom::search_sorted(this->col, ix_1d);
        if(col.get_dtype() != this->col.get_dtype()) col = col.astype(this->row.get_dtype());
        maelstrom::increment(col, DECREMENT); // decrement by 1
        return col;
    }

    maelstrom::vector basic_sparse_matrix::get_values_1d(maelstrom::vector& ix_1d) {
        if(this->val.empty()) return maelstrom::vector();
        return maelstrom::select(this->val, ix_1d);
    }

    maelstrom::vector basic_sparse_matrix::get_relations_1d(maelstrom::vector& ix_1d) {
        if(this->rel.empty()) return maelstrom::vector();
        return maelstrom::select(this->rel, ix_1d);
    }

    std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> basic_sparse_matrix::get_entries_1d(maelstrom::vector& ix_1d) {
        // TODO validity checking of edge ids

        return std::make_tuple(
            std::move(this->get_rows_1d(ix_1d)),
            std::move(this->get_cols_1d(ix_1d)),
            std::move(this->get_values_1d(ix_1d)),
            std::move(this->get_relations_1d(ix_1d))
        );
    }

    maelstrom::vector basic_sparse_matrix::get_1d_index_from_2d_index(maelstrom::vector& ix_r, maelstrom::vector& ix_c, std::any index_not_found) {
        if(!this->sorted) this->sort(); // sort the col ptr

        if(this->format == CSR) {
            return maelstrom::sparse::search_sorted_sparse(this->row, this->col, ix_r, ix_c, index_not_found);
        } else if(this->format == CSC) {
            return maelstrom::sparse::search_sorted_sparse(this->col, this->row, ix_c, ix_r, index_not_found);
        }

        // COO
        throw std::runtime_error("2d-indexing a COO matrix is currently unsupported");        
    }
    
    maelstrom::vector basic_sparse_matrix::get_1d_index_from_value(maelstrom::vector& query_val) {
        auto sort_ix = maelstrom::sort(this->val);
        auto e_index = maelstrom::search_sorted(this->val, query_val);
        maelstrom::increment(e_index, maelstrom::DECREMENT);
        e_index = maelstrom::select(sort_ix, e_index);
        
        // reverse the sort
        sort_ix = maelstrom::sort(sort_ix);
        this->val = maelstrom::select(this->val, sort_ix);
        return e_index;
    }

    std::pair<maelstrom::vector, maelstrom::vector> basic_sparse_matrix::get_values_2d(maelstrom::vector& ix_r, maelstrom::vector& ix_c) {
        auto ix_dtype = this->row.get_dtype();
        auto ix_1d = this->get_1d_index_from_2d_index(ix_r, ix_c, maelstrom::max_value(ix_dtype));

        // drop empties
        auto filter = maelstrom::filter(ix_1d, maelstrom::NOT_EQUALS, maelstrom::max_value(ix_dtype));
        ix_1d = maelstrom::select(ix_1d, filter);
        auto origin = maelstrom::arange(ix_r.get_mem_type(), ix_r.size());
        origin = maelstrom::select(origin, filter);
        filter.clear();

        return std::make_pair(
            std::move(origin),
            std::move(maelstrom::select(this->val, ix_1d))
        );
    }

    std::pair<maelstrom::vector, maelstrom::vector> basic_sparse_matrix::get_relations_2d(maelstrom::vector& ix_r, maelstrom::vector& ix_c) {
        auto ix_dtype = this->row.get_dtype();
        auto ix_1d = this->get_1d_index_from_2d_index(ix_r, ix_c, maelstrom::max_value(ix_dtype));

        // drop empties
        auto filter = maelstrom::filter(ix_1d, maelstrom::NOT_EQUALS, maelstrom::max_value(ix_dtype));
        ix_1d = maelstrom::select(ix_1d, filter);
        auto origin = maelstrom::arange(ix_r.get_mem_type(), ix_r.size());
        origin = maelstrom::select(origin, filter);
        filter.clear();

        return std::make_pair(
            std::move(origin),
            std::move(maelstrom::select(this->rel, ix_1d))
        );
    }

    std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> basic_sparse_matrix::query_adjacency(maelstrom::vector& ix, maelstrom::vector& rel_types, bool return_inner, bool return_values, bool return_relations) {
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

    void basic_sparse_matrix::set(maelstrom::vector new_rows, size_t new_num_rows, maelstrom::vector new_cols, size_t new_num_cols, maelstrom::vector new_vals, maelstrom::vector new_rels) {
        if(this->format != COO) throw std::runtime_error("Can only set for a COO matrix");

        if(new_rows.size() != new_cols.size()) throw std::runtime_error("new rows size must match new cols size");
        if(!new_vals.empty() && new_vals.size() != new_rows.size()) throw std::runtime_error("new vals size must match new rows size");
        if(!new_rels.empty() && new_rels.size() != new_rows.size()) throw std::runtime_error("new rels size must match new rows size");

        if(new_vals.empty() && !this->val.empty()) throw std::runtime_error("values must be inserted since this matrix has values");
        if(new_rels.empty() && !this->rel.empty()) throw std::runtime_error("relations must be inserted since this matrix has values");

        this->n_rows = new_num_rows;
        this->n_cols = new_num_cols;

        if(new_rows.empty()) return;

        this->row.insert(
            this->row.size(),
            new_rows
        );

        this->col.insert(
            this->col.size(),
            new_cols
        );

        if(!new_vals.empty()) {
            this->val.insert(
                this->val.size(),
                new_vals
            );
        }

        if(!new_rels.empty()) {
            this->rel.insert(
                this->rel.size(),
                new_rels
            );
        }

        this->sorted = false;
    }

}