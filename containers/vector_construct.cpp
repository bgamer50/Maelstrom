#include "containers/vector.h"
#include "storage/datatype.h"

namespace maelstrom {

    // Creates a blank vector with the given memory type and data type.
    vector::vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype) {
        this->mem_type = mem_type;
        this->dtype = dtype;
        this->filled_size = 0;
        this->reserved_size = 0;
        this->data_ptr = nullptr;
        this->view = false;
    }

    // Default constructor; creates a blank device vector of FLOAT64 dtype
    vector::vector()
    : vector(maelstrom::storage::DEVICE, maelstrom::float64) {}

    // Creates a vector of size N unitialized values of the given data type and given memory type.
    vector::vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype, size_t N) {
        this->mem_type = mem_type;
        this->dtype = dtype;
        this->filled_size = 0;
        this->reserved_size = 0;
        this->data_ptr = nullptr;
        this->view = false;

        this->resize(N);
    }

    // Creates a vector corresponding to the provided data.  If view=true then this vector is only a view
    // over the provided data.  If view=false then this vector will own a copy of the provided data.
    vector::vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype, void* data, size_t N, bool view) {
        this->mem_type = mem_type;
        this->dtype = dtype;
        this->view = view;
        this->reserved_size = 0;

        if(this->view) { 
            this->data_ptr = data; 
            this->filled_size = N;
            this->reserved_size = N;
        }
        else {
            this->resize(N);
            this->copy(data, this->data_ptr, N);
        }
    }

    vector::vector(vector& orig, bool view)
    : vector(
        orig.mem_type,
        orig.dtype,
        orig.data_ptr,
        orig.filled_size,
        view
    ) {}

    vector::vector(vector& orig) {
        if(&orig == this) {
            return;
        }

        this->mem_type = orig.mem_type;
        this->dtype = orig.dtype;
        this->filled_size = orig.filled_size;
        this->reserved_size = 0;
        this->view = false;

        this->resize(orig.filled_size);
        this->copy(orig.data_ptr, this->data_ptr, orig.filled_size);                    
    }

    vector::~vector() {
        if(this->data_ptr != nullptr && !this->view) {
            this->dealloc(this->data_ptr);
        }
    }

    vector::vector(vector&& other) noexcept {
        if(&other == this) {
            return;
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

        other.data_ptr = nullptr;
        other.filled_size = 0;
        other.reserved_size = 0;
        other.view = true;
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

        other.data_ptr = nullptr;
        other.filled_size = 0;
        other.reserved_size = 0;
        other.view = true;

        return *this;
    }

}