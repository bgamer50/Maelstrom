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
                Returns the format of this matrix. 
            */
            virtual sparse_matrix_format get_format() = 0;

            /*
                Gets the entries corresponding to the 1d index
                Returns (rows, cols, vals).  Vals may be an empty vector
                if this matrix has no values (adjacency matrix).
            */
            virtual std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector> get_entries_1d(maelstrom::vector& ix_1d) = 0;

            /*
                Gets the entries corresponding to the 2d index
            */
            virtual maelstrom::vector get_entries_2d(maelstrom::vector& ix_r, maelstrom::vector& ix_c) = 0;

            /*
                Sets (row, col) = val for each row/col/val in rows/cols/vals.
                If vals is not provided, just sets the row/col (adjacency matrix).
            */
            virtual void set(maelstrom::vector& rows, maelstrom::vector& cols, std::optional<maelstrom::vector&> vals=std::nullopt) = 0;
    };

    class basic_sparse_matrix: public sparse_matrix {
        private:
            maelstrom::vector row;
            maelstrom::vector col;
            maelstrom::vector val;

            maelstrom::sparse_matrix_format format;

            size_t n_rows;
            size_t n_cols;
        
        public:
            basic_sparse_matrix(maelstrom::vector row, maelstrom::vector col, maelstrom::vector values, maelstrom::sparse_matrix_format format);

            using sparse_matrix::has_values;
            inline virtual bool has_values() {
                return !this->val.empty();
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

            using sparse_matrix::get_format;
            inline virtual sparse_matrix_format get_format() { return this->format; }

            using sparse_matrix::get_entries_1d;
            virtual std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector> get_entries_1d(maelstrom::vector& ix_1d);

            using sparse_matrix::get_entries_2d;
            virtual maelstrom::vector get_entries_2d(maelstrom::vector& ix_r, maelstrom::vector& ix_c);

            using sparse_matrix::set;
            virtual void set(maelstrom::vector& rows, maelstrom::vector& cols, std::optional<maelstrom::vector&> vals=std::nullopt);
    };

}