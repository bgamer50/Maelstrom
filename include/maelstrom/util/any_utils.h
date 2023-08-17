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

    template <typename T>
    inline std::vector<T> any_vec_to_vec(std::vector<std::any> any_vec) {
        std::vector<T> new_vec;
        new_vec.reserve(any_vec.size());
        for(std::any& a : any_vec) {
            try {
                new_vec.push_back(
                    std::any_cast<T>(a)
                );
            } catch(std::exception& err) {
                throw std::runtime_error(
                    "1 or more elements of the given vector of anys can't be converted to the specified data type"
                );
            }
        }

        return new_vec;
    }

}