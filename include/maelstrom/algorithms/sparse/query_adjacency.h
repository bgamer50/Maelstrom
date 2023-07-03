#pragma once

#include "maelstrom/containers/sparse_matrix.h"
#include "maelstrom/containers/vector.h"

namespace maelstrom {
    namespace sparse {

        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency(maelstrom::vector& row,
                                                                                                               maelstrom::vector& col,
                                                                                                               maelstrom::vector& val,
                                                                                                               maelstrom::vector& rel,
                                                                                                               maelstrom::vector& ix,
                                                                                                               maelstrom::vector& rel_types,
                                                                                                               bool return_inner,
                                                                                                               bool return_values,
                                                                                                               bool return_relations);

    }
}