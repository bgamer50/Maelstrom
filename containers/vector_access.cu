#include "containers/vector.h"

#include <vector>
#include <cuda_runtime.h>

namespace maelstrom {

    boost::any vector::get(size_t i) {
        size_t data_size = maelstrom::size_of(this->dtype);
        std::vector<unsigned char> raw_value(data_size);
        void* ptr = raw_value.data();

        if(this->mem_type == maelstrom::storage::DEVICE) {
            cudaMemcpy(
                ptr,
                static_cast<unsigned char*>(this->data_ptr) + (data_size * i),
                data_size,
                cudaMemcpyDefault
            );
        } else {
            ptr = static_cast<unsigned char*>(this->data_ptr) + (data_size * i);
        }

        return this->dtype.deserialize(ptr);
    }

    maelstrom::vector as_host_vector(maelstrom::vector& vec) {
        if(vec.get_mem_type() == maelstrom::storage::HOST) {
            return maelstrom::vector(vec, true);
        }

        return vec.to(maelstrom::storage::HOST);
    }

    maelstrom::vector as_device_vector(maelstrom::vector& vec) {
        if(vec.get_mem_type() == maelstrom::storage::DEVICE) {
            return maelstrom::vector(vec, true);
        }

        return vec.to(maelstrom::storage::DEVICE);
    }

}