#include "containers/vector.h"
#include <iostream>

namespace maelstrom {
    
    void vector::clear() {
        if(this->view) throw std::runtime_error("Cannot clear a view!");

        this->dealloc(this->data_ptr);
        this->filled_size = 0;
        this->reserved_size = 0;
        this->data_ptr = nullptr;
    }

    void vector::push_back() {
        throw std::runtime_error("push_back unimplemented");
    }

    void vector::reserve(size_t N) {
        if(N < this->filled_size) throw std::runtime_error("Cannot reserve fewer elements than size!");
        
        if(N <= this->reserved_size) return;

        size_t old_filled_size = this->filled_size;
        this->resize(N);
        this->filled_size = old_filled_size;
    }

    void vector::insert() {
        throw std::runtime_error("insert unimplemented");
    }

    void vector::insert(size_t ix_start, vector& new_elements) {
        if(this->view) throw std::runtime_error("Cannot insert into a view!");
        if(this->data_ptr == new_elements.data_ptr) throw std::runtime_error("Inserted vector cannot be same vector!");

        if(this->dtype.prim_type != new_elements.dtype.prim_type) throw std::runtime_error("Data type of inserting vector must match!");
        

        size_t old_size = this->size();
        size_t new_size = old_size + new_elements.size();
        
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
                static_cast<char*>(new_data) + (element_size * (ix_start + new_elements.size())),
                elements_to_copy
            );
        }

        this->copy(
            new_elements.data(),
            static_cast<char*>(new_data) + (element_size * ix_start),
            new_elements.size()
        );

        if(this->data_ptr != new_data) {
            if(ix_start > 0) {
                this->copy(
                    this->data_ptr,
                    new_data,
                    ix_start
                );
            }

            this->dealloc(this->data_ptr);
            this->data_ptr = new_data;
        }

        this->filled_size = new_size;
    }

    void vector::resize(size_t N) {
        if(this->view) throw std::runtime_error("Cannot resize a view!");

        bool empty = (this->reserved_size == 0);
        
        // Don't resize if there is already enough space reserved
        if(N <= reserved_size) {
            this->filled_size = N;
            return;
        }

        void* new_data = this->alloc(N);
        if(!empty) {
            this->copy(this->data_ptr, new_data, this->filled_size);
            this->dealloc(this->data_ptr);
        }
        
        this->data_ptr = new_data;
        this->filled_size = N;
        this->reserved_size = N;
    }

    vector vector::to(maelstrom::storage new_mem_type) {
        auto new_vec = vector(
            new_mem_type,
            this->get_dtype()
        );

        auto this_view = maelstrom::vector(
            this->mem_type,
            this->dtype,
            this->data_ptr,
            this->filled_size,
            true
        );
        new_vec.insert(0, this_view);
        return new_vec;
    }

}