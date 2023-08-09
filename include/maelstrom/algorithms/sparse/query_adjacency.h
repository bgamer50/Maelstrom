#pragma once

#include "maelstrom/containers/sparse_matrix.h"
#include "maelstrom/containers/vector.h"

namespace maelstrom {
    namespace sparse {

        /*
            Performs an adjacency query, given the row/col vectors of a sparse CSR matrix, the optional
            values array (or empty vector), the optional relations array (or empty vector), the query index
            (which should correspond to row), and optional query relation types (or empty vector), this
            function returns a tuple of vectors (idx, inner, val, rel).

            Accepts flags to return inner index, return values, and return relations.
            By default, return_inner is true, return values is false and return relations is false.
            Return types are (uint64, row/col dtype, val dtype, rel dtype)
        */
        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency(maelstrom::vector& row,
                                                                                                               maelstrom::vector& col,
                                                                                                               maelstrom::vector& val,
                                                                                                               maelstrom::vector& rel,
                                                                                                               maelstrom::vector& ix,
                                                                                                               maelstrom::vector& rel_types,
                                                                                                               bool return_inner=true,
                                                                                                               bool return_values=false,
                                                                                                               bool return_relations=false,
                                                                                                               bool return_1d_index_as_values=false);

    }
}