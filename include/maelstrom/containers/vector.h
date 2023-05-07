#pragma once

#include <cstdint>
#include <inttypes.h>
#include <string>

#include "storage/storage.h"
#include "storage/datatype.h"

namespace maelstrom {

    class vector {
        private:
            void* data_ptr;
            
            size_t filled_size;
            size_t reserved_size;
            
            maelstrom::storage mem_type;
            maelstrom::dtype_t dtype;
            
            bool view;

            void* alloc(size_t N);

            void dealloc(void* ptr);

            void copy(void* src, void* dst, size_t size);
        
        public:
            std::string name;

            // Creates a blank vector with the given memory type and data type.
            vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype);

            // Default constructor; creates a blank device vector of FLOAT64 dtype
            vector();

            // Creates a vector of size N unitialized values of the given data type and given memory type.
            vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype, size_t N);

            // Creates a vector corresponding to the provided data.  If view=true then this vector is only a view
            // over the provided data.  If view=false then this vector will own a copy of the provided data.
            vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype, void* data, size_t N, bool view=true);

            vector(vector& orig, bool view);

            vector(vector& orig);

            ~vector();

            vector(vector&& other) noexcept;
            
            vector& operator=(vector&& other) noexcept;

            inline bool is_view() { return this->view; }

            void push_back();

            void reserve(size_t N);

            void insert(); // single insert, range insert

            void insert(size_t ix_start, vector& new_elements);

            inline void erase(){} // single erase, range erase
            inline void get(){} // single get, range get

            /*
                Empties the contents of this vector and frees any reserved memory.
            */
            void clear();

            /*
                Copies the vector to the host and prints it.
            */
            void print();
            
            inline size_t size() { return this->filled_size; }

            inline void* data() {
                return this->data_ptr;
            }

            inline maelstrom::dtype_t get_dtype() { return this->dtype; }

            inline maelstrom::storage get_mem_type() { return this->mem_type; }

            // This currently-viewing vector will take ownership of the data and it will
            // follow the lifecycle of this vector.
            inline void own() {
                if(this->view) this->view = false;
                else throw std::runtime_error("Vector already owns data!"); 
            }

            /*
                The currently-owning vector will no longer own its data, and it will
                not be cleaned up when this vector is deleted.  Essentially, this
                vector becomes a view.
            */
            inline void disown() {
                if(this->view) throw std::runtime_error("Vector does not own data!");
                else this->view = true;
            }

            void resize(size_t N);

            /*
                Creates a copy of this vector with the given memory type.
            */
            vector to(maelstrom::storage mem_type);

    };

}