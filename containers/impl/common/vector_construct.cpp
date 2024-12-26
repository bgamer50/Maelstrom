#include "maelstrom/containers/vector.h"
#include "maelstrom/storage/datatype.h"
#include <sstream>
#include <iostream>

namespace maelstrom {

    // Creates a blank vector with the given memory type and data type.
    vector::vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype) {
        this->mem_type = mem_type;
        this->dtype = dtype;
        this->filled_size = 0;
        this->reserved_size = 0;
        this->data_ptr = nullptr;
        this->view = false;
        this->stream = get_default_stream(mem_type);
    }

    // Default constructor; creates a blank device vector of default dtype
    vector::vector()
    : vector(maelstrom::storage::DEVICE, maelstrom::default_dtype) {}

    // Creates a vector of global_size unitialized values of the given data type and given memory type.
    vector::vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype, size_t global_size) {
        this->mem_type = mem_type;
        this->dtype = dtype;
        this->filled_size = 0;
        this->reserved_size = 0;
        this->data_ptr = nullptr;
        this->view = false;
        this->stream = get_default_stream(mem_type);

        this->resize(global_size);
        
    }

    // Creates a vector of global_size unitialized values of the give data type and given memory type
    // with the given local partion size.  Verifies that the partitions are legal.
    vector::vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype, size_t global_size, size_t local_partition_size) {
        this->mem_type = mem_type;
        this->dtype = dtype;
        this->filled_size = 0;
        this->reserved_size = 0;
        this->stream = get_default_stream(mem_type);
        this->view = false;
        this->data_ptr = nullptr;

        this->resize_local(local_partition_size);
        if(this->size() != global_size) {
            throw std::invalid_argument("Invalid partition specification!");
        }
    }

    // Creates a vector corresponding to the provided data.  If view=true then this vector is only a view
    // over the provided data.  If view=false then this vector will own a copy of the provided data.
    vector::vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype, void* data, size_t local_partition_size, bool view) {
        this->mem_type = mem_type;
        this->dtype = dtype;
        this->view = view;
        this->reserved_size = 0;
        this->filled_size = 0;
        this->stream = get_default_stream(mem_type);

        if(this->view) { 
            this->data_ptr = data; 
            this->filled_size = local_partition_size;
            this->reserved_size = local_partition_size;
        }
        else {
            this->data_ptr = nullptr;
            this->resize_local(local_partition_size);
            this->copy(data, this->data_ptr, local_partition_size);
        }
    }

    vector::vector(const vector& orig, bool view) {
        // can't implement this as another constructor call
        // since orig is const and we must preserve the original
        // partitions if the vector is distributed
        this->mem_type = orig.mem_type;
        this->dtype = orig.dtype;
        this->filled_size = 0;
        this->reserved_size = 0;
        this->view = view;
        this->stream = orig.stream;

        if(view) {
            this->data_ptr = orig.data_ptr;
            this->filled_size = orig.filled_size;
            this->reserved_size = orig.filled_size;
        } else {
            // copy
            this->data_ptr = nullptr;
            if(orig.filled_size > 0) {
                this->resize_local(orig.filled_size);
                this->copy(orig.data_ptr, this->data_ptr, orig.filled_size);
            }
        }
    }

    vector::vector(const vector& orig) {
        if(&orig == this) {
            return;
        }

        this->mem_type = orig.mem_type;
        this->dtype = orig.dtype;
        this->filled_size = orig.filled_size;
        this->reserved_size = 0;
        this->view = false;
        this->data_ptr = nullptr;
        this->stream = orig.stream;

        if(this->filled_size > 0) {
            this->resize_local(orig.filled_size);
            this->copy(orig.data_ptr, this->data_ptr, orig.filled_size);                    
        }
    }

    vector::vector(vector&& other) noexcept {
        if(&other == this) {
            return;
        }

        this->data_ptr = other.data_ptr;
        this->filled_size = other.filled_size;
        this->reserved_size = other.reserved_size;
        this->dtype = other.dtype;
        this->mem_type = other.mem_type;
        this->view = other.view;
        this->stream = other.stream;

        other.data_ptr = nullptr;
        other.filled_size = 0;
        other.reserved_size = 0;
        other.view = true;
    }

    vector::~vector() {
        if(this->data_ptr != nullptr && !this->view) {
            this->dealloc(this->data_ptr);
        }
    }
    
    vector& vector::operator=(vector&& other) noexcept {
        if(&other == this) {
            return *this;
        }

        if(!this->view && this->reserved_size > 0) {
            this->clear();
        }

        this->data_ptr = other.data_ptr;
        this->filled_size = other.filled_size;
        this->reserved_size = other.reserved_size;
        this->dtype = other.dtype;
        this->mem_type = other.mem_type;
        this->view = other.view;
        this->stream = other.stream;

        other.data_ptr = nullptr;
        other.filled_size = 0;
        other.reserved_size = 0;
        other.view = true;

        return *this;
    }

    vector& vector::operator=(const vector& other) noexcept {
        if(&other == this) {
            return *this;
        }

        if(!this->view) {
            this->clear();
        }

        this->mem_type = other.mem_type;
        this->dtype = other.dtype;
        this->view = other.view;
        this->stream = other.stream;

        this->data_ptr = this->alloc(other.reserved_size);
        this->filled_size = other.filled_size;
        this->reserved_size = other.reserved_size;
        this->copy(other.data_ptr, this->data_ptr, this->reserved_size);

        return *this;
    }

    vector make_vector_from_anys(maelstrom::storage mem_type, maelstrom::dtype_t dtype, std::vector<std::any>& anys) {
        if(anys.empty()) return maelstrom::vector(mem_type, dtype);

        std::vector<unsigned char> bytes;
        bytes.reserve(anys.size() * maelstrom::size_of(dtype));
        for(size_t k = 0; k < anys.size(); ++k) {
            std::vector<unsigned char> bytes_k;
            maelstrom::primitive_t prim_k;
            std::tie(bytes_k, prim_k) = maelstrom::any_to_bytes(anys[k]);
            
            if(prim_k != dtype.prim_type) {
                std::stringstream sx;
                sx << "Type mismatch in array at index " << k;
                throw std::runtime_error(sx.str());
            }

            bytes.insert(bytes.end(), bytes_k.begin(), bytes_k.end());
            
        }

        return maelstrom::vector(
            mem_type,
            dtype,
            bytes.data(),
            anys.size(),
            false
        );
    }

    vector make_vector_from_anys(maelstrom::storage mem_type, std::vector<std::any>& anys) {
        if(anys.empty()) return maelstrom::vector(mem_type, maelstrom::default_dtype);
        maelstrom::dtype_t dtype = maelstrom::dtype_from_prim_type(
            maelstrom::prim_type_of(anys.front())
        );

        return make_vector_from_anys(
            mem_type,
            dtype,
            anys
        );
    }

}