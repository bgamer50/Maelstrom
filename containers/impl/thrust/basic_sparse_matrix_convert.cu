#include "maelstrom/containers/sparse_matrix.h"
#include "maelstrom/algorithms/sort.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/sparse/csr_to_coo.h"
#include "maelstrom/algorithms/reduce_by_key.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/prefix_sum.h"

namespace maelstrom {
    void basic_sparse_matrix::sort() {
        if(this->sorted) return;

        if(this->format != COO) throw std::runtime_error("Sorting CSR/CSC is currently unsupported!");

        auto sorted_ix = maelstrom::sort({
            std::ref(this->row),
            std::ref(this->col)
        });
        
        if(this->has_values()) this->val = std::move(maelstrom::select(this->val, sorted_ix));
        if(this->has_relations()) this->rel = std::move(maelstrom::select(this->rel, sorted_ix));
        
        sorted_ix.clear();
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
        maelstrom::set(vals, 1);

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
        auto sort_ix = maelstrom::sort(output_indices);
        output_indices.clear();

        output_counts = maelstrom::select(output_counts, sort_ix);
        sort_ix.clear();

        maelstrom::vector first_zero(output_counts.get_mem_type(), output_counts.get_dtype(), 1);
        maelstrom::set(first_zero, 0);
        output_counts.insert(0, first_zero);
        first_zero.clear();
        maelstrom::prefix_sum(output_counts);

        this->row = std::move(output_counts);
        this->format = CSR;
    }

    void basic_sparse_matrix::to_csc() {
        if(this->format == CSC) {
            return;
        } else if(this->format == CSR) {
            this->to_coo();
        }

        std::swap(this->col, this->row);
        this->sorted = false;

        this->to_csr();
        std::swap(this->col, this->row);
        this->format = CSC;
    }
}