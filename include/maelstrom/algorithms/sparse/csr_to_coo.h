#pragma once
#include "maelstrom/containers/vector.h"

namespace maelstrom {
    namespace sparse {

        /*
            Given a CSR row pointer, returns a COO
            row index.
        */
        maelstrom::vector csr_to_coo(maelstrom::vector& ptr, size_t nnz);

    }
}