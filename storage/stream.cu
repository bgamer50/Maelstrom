#include "maelstrom/storage/storage.h"
#include <cuda_runtime.h>

namespace maelstrom {

    std::any create_stream(storage s) {
        auto t = maelstrom::single_storage_of(s);
        switch(t) {
            case DEVICE:
            case MANAGED: {
                cudaStream_t st;
                cudaStreamCreate(&st);
                return st;
            }
            case PINNED:
            case HOST: {
                return std::any();
            }
        }

        throw std::invalid_argument("Invalid storage");
    }

    void destroy_stream(storage s, std::any stream) {
        cudaStreamDestroy(
            std::any_cast<cudaStream_t>(stream)
        );
    }

    std::any get_default_stream(storage s) { return (cudaStream_t)cudaStreamDefault; }

}