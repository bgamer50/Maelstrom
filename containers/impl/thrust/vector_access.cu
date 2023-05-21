#include "containers/vector.h"

#include <vector>
#include <cuda_runtime.h>

namespace maelstrom {

    boost::any vector::get(size_t i) {
        if(i >= this->filled_size) throw std::out_of_range("Attempted to get element out of bounds!");

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

    maelstrom::vector as_primitive_vector(maelstrom::vector& vec, bool view) {
        return maelstrom::vector(
            vec.get_mem_type(),
            maelstrom::dtype_from_prim_type(vec.get_dtype().prim_type),
            vec.data(),
            vec.size(),
            view
        );
    }

}