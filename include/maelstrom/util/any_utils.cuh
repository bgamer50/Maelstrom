#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {

    inline std::any safe_any_cast(std::any b, maelstrom::dtype_t new_dtype) {
        try {
            return new_dtype.serialize(b);
        } catch(std::exception& err) { // TODO change this to a different maelstrom-specific exception
            std::vector<std::any> anys = {b};
            auto b_vec = maelstrom::make_vector_from_anys(maelstrom::HOST, anys).astype(new_dtype);
            return b_vec.get(0);
        }
    }

}