#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/cast.h"

namespace maelstrom {
    
    void vector::clear() {
        if(this->view) throw std::runtime_error("Cannot clear a view!");
        if(this->reserved_size == 0 || this->data_ptr == nullptr) return;

        this->dealloc(this->data_ptr);
        this->filled_size = 0;
        this->reserved_size = 0;
        this->data_ptr = nullptr;
    }

    void vector::push_back() {
        throw std::runtime_error("push_back unimplemented");
    }

    void vector::insert_local(size_t ix_start, vector& new_elements, size_t add_ix_start, size_t add_ix_end) {
        if(new_elements.filled_size == 0) {
            if(add_ix_end - add_ix_start > 0) throw std::runtime_error("Invalid range in new elements (empty vector)");
            else return;
        }

        if(this->view) throw std::runtime_error("Cannot insert into a view!");
        if(this->data_ptr == new_elements.data_ptr) throw std::runtime_error("Inserted vector cannot be same vector!");

        if(this->dtype.prim_type != new_elements.dtype.prim_type) throw std::runtime_error("Data type of inserting vector must match!");

        if(add_ix_end < add_ix_start) throw std::runtime_error("Invalid range in new elements");
        
        size_t insert_size = add_ix_end - add_ix_start;
        size_t old_size = this->local_size();
        size_t new_size = old_size + insert_size;
        
        void* new_data = this->data_ptr;
        if(new_size > reserved_size) {
            new_data = this->alloc(new_size);
            this->reserved_size = new_size;   
        }

        size_t elements_to_copy = old_size - ix_start;
        size_t element_size = maelstrom::size_of(this->dtype);

        if(elements_to_copy > 0) {
            this->copy(
                static_cast<char*>(this->data_ptr) + (element_size * ix_start),
                static_cast<char*>(new_data) + (element_size * (ix_start + insert_size)),
                elements_to_copy
            );
        }

        this->copy(
            static_cast<unsigned char*>(new_elements.data()) + element_size * add_ix_start,
            static_cast<unsigned char*>(new_data) + (element_size * ix_start),
            insert_size
        );

        if(this->data_ptr != new_data) {
            if(ix_start > 0) {
                this->copy(
                    this->data_ptr,
                    new_data,
                    ix_start
                );
            }

            if(this->data_ptr != nullptr) {
                this->dealloc(this->data_ptr);
            }
            this->data_ptr = new_data;
        }

        this->filled_size = new_size;
    }

    void vector::insert(size_t ix_start, vector& new_elements, size_t add_ix_start, size_t add_ix_end) {
        if(maelstrom::is_dist(this->mem_type)) {
            throw std::runtime_error("Cannot insert into distributed vector!");
        }

        return this->insert_local(
            ix_start,
            new_elements,
            add_ix_start,
            add_ix_end
        );
    }

    void vector::shrink_to_fit() {
        if(this->view) throw std::runtime_error("Cannot shrink a view!");
        if(this->empty() || this->filled_size == this->reserved_size) return;

        void* new_data = this->alloc(this->filled_size);
        this->copy(
            this->data_ptr,
            new_data,
            this->filled_size
        );

        this->dealloc(this->data_ptr);
        this->data_ptr = new_data;
        this->reserved_size = this->filled_size;
    }

    vector vector::astype(maelstrom::dtype_t new_dtype) {
        return maelstrom::cast(*this, new_dtype);
    }

    vector vector::to(maelstrom::storage new_mem_type) {
        auto new_vec = vector(
            new_mem_type,
            this->get_dtype()
        );

        auto this_view = maelstrom::vector(
            maelstrom::single_storage_of(this->mem_type),
            this->dtype,
            this->data_ptr,
            this->filled_size,
            true
        );
        new_vec.insert_local(0, this_view);
        return new_vec;
    }

}