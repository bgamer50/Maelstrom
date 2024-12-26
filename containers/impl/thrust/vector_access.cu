#include "maelstrom/containers/vector.h"
#include "maelstrom/dist_utils/dist_partition.cuh"

#include <vector>
#include <sstream>
#include <numeric>

#include <iostream>

#include <cuda_runtime.h>
#include "nccl.h"

#include "maelstrom/util/cuda_utils.cuh"
#include "maelstrom/dist_utils/nccl_utils.cuh"
#include "maelstrom/algorithms/prefix_sum.h"
#include "maelstrom/algorithms/search_sorted.h"

namespace maelstrom {

    std::any vector::get_local(size_t i) {
        if(i >= this->filled_size) throw std::out_of_range("Attempted to get element out of bounds!");

        size_t data_size = maelstrom::size_of(this->dtype);
        std::vector<unsigned char> raw_value(data_size);
        void* ptr = raw_value.data();

        auto single_mem_type = maelstrom::single_storage_of(this->mem_type);
        if(single_mem_type == maelstrom::storage::DEVICE) {
            cudaMemcpy(
                ptr,
                static_cast<unsigned char*>(this->data_ptr) + (data_size * i),
                data_size,
                cudaMemcpyDefault
            );
        } else {
            ptr = static_cast<unsigned char*>(this->data_ptr) + (data_size * i);
        }

        this->dtype.deserialize(ptr);
        return this->dtype.deserialize(ptr);
    }

    std::any vector::get(size_t i) {
        if(maelstrom::is_dist(this->mem_type)) {
            throw std::runtime_error("Calling get() is not supported in distributed processing.  Did you intend to call get_local()?");
        }
        
        return get_local(i);
    }

    void vector::erase(size_t i) {
        if(this->filled_size == 0) {
            throw std::out_of_range("Can't erase elements of an empty vector");
        }

        if(i >= this->filled_size) {
            std::stringstream sx;
            sx << "Element " << i << " is out of range for vector of size " << this->filled_size;
            throw std::out_of_range(sx.str());
        }

        if(i == 0 && this->filled_size == 1) {
            return this->clear();
        }

        auto data_size = maelstrom::size_of(this->dtype);

        auto erase_ptr = static_cast<unsigned char*>(this->data_ptr) + (data_size * (i+1));
        auto new_data = static_cast<unsigned char*>(this->alloc(this->reserved_size));

        if(i > 0) {
            this->copy(this->data_ptr, new_data, i-1);
        }
        if(i < this->filled_size - 1) {
            this->copy(erase_ptr, new_data + (data_size * i), this->filled_size - i - 1);
        }

        this->dealloc(this->data_ptr);
        this->data_ptr = new_data;

        this->filled_size -= 1;
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
        auto prim_dtype = maelstrom::dtype_from_prim_type(vec.get_dtype().prim_type);

        if(vec.size() == 0) {
            return maelstrom::vector(
                vec.get_mem_type(),
                vec.get_dtype()
            );
        }

        return maelstrom::vector(
            vec.get_mem_type(),
            prim_dtype,
            vec.data(),
            vec.size(),
            view
        );
    }

}