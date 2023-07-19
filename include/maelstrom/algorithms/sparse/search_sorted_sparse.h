#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {
    namespace sparse { 

        maelstrom::vector search_sorted_sparse(maelstrom::vector& row, maelstrom::vector& col, maelstrom::vector& ix_r, maelstrom::vector& ix_c, std::any index_not_found=std::any());

    }
}