#pragma once

#include "maelstrom/containers/vector.h"

#include <optional>

namespace maelstrom {

    enum sparse_matrix_format {
        COO = 0,
        CSR = 1,
        CSC = 2
    };

    class sparse_matrix {
        public:
            /*
                Returns true if this matrix has values, false otherwise.
            */
            virtual bool has_values() = 0;

            /*
                Returns true if this matrix has relations, false otherwise.
            */
            virtual bool has_relations() = 0;

            /*
                Returns the number of rows in the matrix.
            */
            virtual size_t num_rows() = 0;

            /*
                Returns the number of columns in the matrix.
            */
            virtual size_t num_cols() = 0;

            /*
                Returns the number of nonzero entries in the matrix.
            */
            virtual size_t num_nonzero() = 0;

            /*
                Returns true if this matrix is sorted (row then col for COO,
                col for CSR, row for CSC), otherwise returns false.
            */
            virtual bool is_sorted() = 0;

            /*
                Sorts this sparse matrix (row then col for COO, col for CSR,
                row for CSC).
                If return_perm is true, and this is a COO matrix, returns the permutation
                from sorting this matrix.
                If return_perm is false, or this is not a COO matrix, an empty vector
                is returned.
            */
            virtual maelstrom::vector sort(bool return_perm=false) = 0;

            /*
                Returns the format of this matrix. 
            */
            virtual sparse_matrix_format get_format() = 0;

            /*
                Gets the entries corresponding to the 1d index
                Returns (rows, cols, vals, relations).  Vals may be an empty vector
                if this matrix has no values (adjacency matrix).  Relations may
                be empty if this matrix has no relations.
            */
            virtual std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> get_entries_1d(maelstrom::vector& ix_1d) = 0;

            /*
                Gets the rows corresponding to the 1d index
            */
            virtual maelstrom::vector get_rows_1d(maelstrom::vector& ix_1d) = 0;

            /*
                Gets the columns corresponding to the 1d index
            */
            virtual maelstrom::vector get_cols_1d(maelstrom::vector& ix_1d) = 0;

            /*
                Gets the values corresponding to the 1d index,
                can be empty if there are no values (sparse matrix)
            */
            virtual maelstrom::vector get_values_1d(maelstrom::vector& ix_1d) = 0;

            /*
                Gets the values corresponding to the 2d index
                For matrices with multiple values for the same index
                (i.e. a matrix representing a multi-graph), this will
                only return one arbitrary value, which is not guaranteed
                to be the same value each time.

                Returns originating indices, values
            */
            virtual std::pair<maelstrom::vector, maelstrom::vector> get_values_2d(maelstrom::vector& ix_r, maelstrom::vector& ix_c) = 0;

            virtual maelstrom::vector get_relations_1d(maelstrom::vector& ix_1d) = 0;

            /*
                Returns originating indices, relations
            */
            virtual std::pair<maelstrom::vector, maelstrom::vector> get_relations_2d(maelstrom::vector& ix_r, maelstrom::vector& ix_c) = 0;

            virtual maelstrom::vector get_1d_index_from_2d_index(maelstrom::vector& ix_r, maelstrom::vector& ix_c, std::any index_not_found=std::any()) = 0;

            virtual maelstrom::vector get_1d_index_from_value(maelstrom::vector& query_val) = 0;

            /*
                Depending on the format of this matrix, this operation is slightly different.
                
                For CSR, gets the nonzero columns and values.  For a matrix with multiple values
                for the same index, it will return all values.

                For CSC, gets the nonzero rows and values.  For a matrix with multiple values for
                the same index, it will return all values.

                For COO, this operation is invalid and will throw an exception.

                In all cases, returns a tuple of (original indices, rows/cols, values, relations).
                Values can be an empty vector if this matrix has no values.

                Has flags to return the inner index (col for CSR, row for CSC), values, and relations.
                Defaults to only returning the inner index (return_inner=true, return_values=false, return_relations=false).
            */
            virtual std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency(maelstrom::vector& ix, maelstrom::vector& rel_types, bool return_inner=true, bool return_values=false, bool return_relations=false) = 0;

            /*
                Sets (row, col) = val for each row/col/val in rows/cols/vals.
                If vals is not provided, just sets the row/col (adjacency matrix).
                Can optionally set relation too.
            */
            virtual void set(maelstrom::vector new_rows, size_t new_num_rows, maelstrom::vector new_cols, size_t new_num_cols, maelstrom::vector new_vals=maelstrom::vector(), maelstrom::vector new_rels=maelstrom::vector()) = 0;

            virtual void to_csr() = 0;
            virtual void to_csc() = 0;
            virtual void to_coo() = 0;

            virtual maelstrom::vector get_row() = 0;
            virtual maelstrom::vector get_col() = 0;
            virtual maelstrom::vector get_val() = 0;
            virtual maelstrom::vector get_rel() = 0;
    };

    class basic_sparse_matrix: public sparse_matrix {
        private:
            maelstrom::vector row;
            maelstrom::vector col;
            maelstrom::vector val;
            maelstrom::vector rel;

            maelstrom::sparse_matrix_format format;

            size_t n_rows;
            size_t n_cols;

            bool sorted;
        
        public:
            inline basic_sparse_matrix(maelstrom::vector rows, maelstrom::vector cols, maelstrom::vector values, maelstrom::vector relations, maelstrom::sparse_matrix_format fmt, size_t num_rows, size_t num_cols, bool sorted=false) {
                this->row = std::move(rows);
                this->col = std::move(cols);
                this->val = std::move(values);
                this->rel = std::move(relations);

                this->n_rows = num_rows;
                this->n_cols = num_cols;
                this->format = fmt;
                this->sorted = sorted;
            }

            using sparse_matrix::has_values;
            inline virtual bool has_values() {
                return !this->val.empty();
            }

            using sparse_matrix::has_relations;
            inline virtual bool has_relations() {
                return !this->rel.empty();
            }

            using sparse_matrix::num_rows;
            inline virtual size_t num_rows() { return this->n_rows; }

            using sparse_matrix::num_cols;
            inline virtual size_t num_cols() { return this->n_cols; }

            using sparse_matrix::num_nonzero;
            inline virtual size_t num_nonzero() {
                switch(this->format) {
                    case COO:
                        return this->row.size();
                    case CSR:
                        return this->col.size();
                    case CSC:
                        return this->row.size();
                }

                throw std::runtime_error("Invalid matrix format");
            }

            using sparse_matrix::is_sorted;
            inline virtual bool is_sorted() { return this->sorted; }

            using sparse_matrix::sort;
            virtual maelstrom::vector sort(bool return_perm=false);

            using sparse_matrix::get_format;
            inline virtual sparse_matrix_format get_format() { return this->format; }

            using sparse_matrix::get_entries_1d;
            virtual std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> get_entries_1d(maelstrom::vector& ix_1d);

            using sparse_matrix::get_rows_1d;
            virtual maelstrom::vector get_rows_1d(maelstrom::vector& ix_1d);

            using sparse_matrix::get_cols_1d;
            virtual maelstrom::vector get_cols_1d(maelstrom::vector& ix_1d);

            using sparse_matrix::get_values_1d;
            virtual maelstrom::vector get_values_1d(maelstrom::vector& ix_1d);

            using sparse_matrix::get_values_2d;
            virtual std::pair<maelstrom::vector, maelstrom::vector> get_values_2d(maelstrom::vector& ix_r, maelstrom::vector& ix_c);

            using sparse_matrix::get_relations_1d;
            virtual maelstrom::vector get_relations_1d(maelstrom::vector& ix_1d);

            using sparse_matrix::get_relations_2d;
            virtual std::pair<maelstrom::vector, maelstrom::vector> get_relations_2d(maelstrom::vector& ix_r, maelstrom::vector& ix_c);

            using sparse_matrix::get_1d_index_from_2d_index;
            virtual maelstrom::vector get_1d_index_from_2d_index(maelstrom::vector& ix_r, maelstrom::vector& ix_c, std::any index_not_found=std::any());
        
            using sparse_matrix::get_1d_index_from_value;
            virtual maelstrom::vector get_1d_index_from_value(maelstrom::vector& query_val);

            using sparse_matrix::query_adjacency;
            virtual std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency(maelstrom::vector& ix, maelstrom::vector& rel_types, bool return_inner=true, bool return_values=false, bool return_relations=false);

            using sparse_matrix::set;
            virtual void set(maelstrom::vector new_rows, size_t new_num_rows, maelstrom::vector new_cols, size_t new_num_cols, maelstrom::vector new_vals=maelstrom::vector(), maelstrom::vector new_rels=maelstrom::vector());

            using sparse_matrix::to_csr;
            virtual void to_csr();

            using sparse_matrix::to_csc;
            virtual void to_csc();

            using sparse_matrix::to_coo;
            virtual void to_coo();

            using sparse_matrix::get_row;
            inline virtual maelstrom::vector get_row() { return this->row; }
            
            using sparse_matrix::get_col;
            inline virtual maelstrom::vector get_col() { return this->col; }

            using sparse_matrix::get_val;
            inline virtual maelstrom::vector get_val() { return this->val; }

            using sparse_matrix::get_rel;
            inline virtual maelstrom::vector get_rel() { return this->rel; }
    };

}