#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {

    inline boost::any safe_any_cast(boost::any b, maelstrom::dtype_t new_dtype) {
        std::vector<boost::any> anys = {b};
        auto b_vec = maelstrom::make_vector_from_anys(maelstrom::HOST, anys).astype(new_dtype);
        return b_vec.get(0);
    }

}