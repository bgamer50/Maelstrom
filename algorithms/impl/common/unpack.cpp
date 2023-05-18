#include "algorithms/unpack.h"

namespace maelstrom {

    std::vector<maelstrom::vector> unpack(maelstrom::vector& vec) {
        std::vector<maelstrom::vector> unpacked_vector;
        unpacked_vector.resize(vec.size());

        for(size_t k = 0; k < vec.size(); ++k) {
            unpacked_vector[k] = maelstrom::make_vector_like(vec);
            unpacked_vector[k].insert(
                0,
                vec,
                k,
                k + 1
            );
        }

        return unpacked_vector;
    }

}