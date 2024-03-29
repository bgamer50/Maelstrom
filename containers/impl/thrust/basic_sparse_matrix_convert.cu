#include "maelstrom/containers/sparse_matrix.h"
#include "maelstrom/algorithms/sort.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/sparse/csr_to_coo.h"
#include "maelstrom/algorithms/reduce_by_key.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/prefix_sum.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/assign.h"

#include <iostream>

namespace maelstrom {
    maelstrom::vector basic_sparse_matrix::sort(bool return_perm) {
        if(this->sorted) {
            if(return_perm) {
                return maelstrom::arange(this->val.get_mem_type(), this->val.size());
            } else {
                return maelstrom::vector();
            }
        }

        if(this->format != COO) throw std::runtime_error("Sorting CSR/CSC is currently unsupported!");

        auto sorted_ix = maelstrom::sort({
            std::ref(this->row),
            std::ref(this->col)
        });
        
        if(this->has_values()) this->val = std::move(maelstrom::select(this->val, sorted_ix));
        if(this->has_relations()) this->rel = std::move(maelstrom::select(this->rel, sorted_ix));
        
        if(return_perm) return sorted_ix;
        return maelstrom::vector();
    }

    maelstrom::vector basic_sparse_matrix::sort_values(bool return_perm) {
        if(this->format != COO) throw std::runtime_error("Sorting by values is only valid for COO matrices!");

        auto sorted_ix = maelstrom::sort(this->val);
        this->row = std::move(maelstrom::select(this->row, sorted_ix));
        this->col = std::move(maelstrom::select(this->col, sorted_ix));
        if(this->has_relations()) {
            this->rel = std::move(maelstrom::select(this->rel, sorted_ix));
        }

        this->sorted = false;
        return return_perm ? sorted_ix : maelstrom::vector();
    }

    void basic_sparse_matrix::to_coo() {
        if(this->format == COO) {
            return;
        } else if(this->format == CSR) {
            this->row = std::move(
                maelstrom::sparse::csr_to_coo(this->row, this->num_nonzero())
            );  
        } else if(this->format == CSC) {
            this->col = std::move(
                maelstrom::sparse::csr_to_coo(this->col, this->num_nonzero())
            );
            this->sorted = false;
        } else {
            throw std::runtime_error("invalid matrix format");
        }

        this->format = COO;
    }

    void basic_sparse_matrix::to_csr() {
        if(this->format == CSR) {
            return;
        }

        this->to_coo();

        // Convert COO to CSR
        this->sort();

        maelstrom::vector vals(
            this->row.get_mem_type(),
            this->row.get_dtype(),
            this->row.size()
        );
        auto vals_prim_view = maelstrom::as_primitive_vector(vals, true);

        maelstrom::set(
            vals_prim_view,
            1
        );

        maelstrom::vector output_counts;
        maelstrom::vector output_indices;
        std::tie(output_counts, output_indices) = maelstrom::reduce_by_key(
            this->row,
            vals,
            maelstrom::SUM,
            true
        );
        vals.clear();

        output_indices = maelstrom::select(this->row, output_indices);

        auto actual_counts = maelstrom::make_vector_like(output_counts);
        actual_counts.resize(this->n_rows);
        auto actual_counts_prim_view = maelstrom::as_primitive_vector(actual_counts, true);
        maelstrom::set(actual_counts_prim_view, 0);
        
        maelstrom::assign(actual_counts, output_indices, output_counts);
        output_indices.clear();
        output_counts.clear();

        maelstrom::vector first_zero(actual_counts.get_mem_type(), actual_counts.get_dtype(), 1);
        auto first_zero_prim_view = maelstrom::as_primitive_vector(first_zero, true);
        maelstrom::set(first_zero_prim_view, 0);
        actual_counts.insert(0, first_zero);
        first_zero.clear();
        maelstrom::prefix_sum(actual_counts);

        this->row = std::move(actual_counts);
        this->format = CSR;
    }

    void basic_sparse_matrix::to_csc() {
        if(this->format == CSC) {
            return;
        } else if(this->format == CSR) {
            this->to_coo();
        }

        std::swap(this->col, this->row);
        std::swap(this->n_cols, this->n_rows);
        this->sorted = false;

        this->to_csr();
        std::swap(this->col, this->row);
        std::swap(this->n_cols, this->n_rows);
        this->format = CSC;
    }
}