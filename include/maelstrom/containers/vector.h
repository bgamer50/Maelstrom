#pragma once

#include <cstdint>
#include <inttypes.h>
#include <string>
#include <stdexcept>

#include "maelstrom/storage/storage.h"
#include "maelstrom/storage/datatype.h"

namespace maelstrom {

    class vector {
        private:
            void* data_ptr;
            
            size_t filled_size;
            size_t reserved_size;
            
            maelstrom::storage mem_type;
            maelstrom::dtype_t dtype;
            std::any stream;
            
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

            // Creates a vector of global_size unitialized values of the given data type and given memory type.
            vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype, size_t global_size);

            // Safely creates a vector of global_size unitialized values of the given data type and memory type,
            // with the specificed local partition size.
            vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype, size_t global_size, size_t local_partition_size);

            // Creates a vector corresponding to the provided data.  If view=true then this vector is only a view
            // over the provided data.  If view=false then this vector will own a copy of the provided data.
            vector(maelstrom::storage mem_type, maelstrom::dtype_t dtype, void* data, size_t local_partition_size, bool view=true);

            vector(const vector& orig, bool view);

            vector(const vector& orig);

            ~vector();

            vector(vector&& other) noexcept;
            
            vector& operator=(vector&& other) noexcept;

            vector& operator=(const vector& other) noexcept;

            inline bool is_dist() { return maelstrom::is_dist(this->mem_type); }

            inline bool is_view() { return this->view; }

            /*
                Pins the memory viewed by this vector.  If this vector is not a host view,
                an exception is thrown.
            */
            void pin();

            /*
                Unpins the memory viewed by this vector.  If this vector is not a host view,
                an exception is thrown.
            */
            void unpin();

            void push_back();

            void reserve(size_t N);
            void reserve_local(size_t N);

            /*
                At position ix_start in this vector, adds the elements in the given vector
                in the range [add_ix_start, add_ix_end).
            */
            void insert(size_t ix_start, vector& new_elements, size_t add_ix_start, size_t add_ix_end);
            void insert_local(size_t ix_start, vector& new_elements, size_t add_ix_start, size_t add_ix_end);

            /*
                At position ix_start in this vector, add all elements in the given vector.
            */
            inline void insert(size_t ix_start, vector& new_elements) {
                return this->insert(ix_start, new_elements, 0, new_elements.size());
            }
            inline void insert_local(size_t ix_start, vector& new_elements) {
                return this->insert_local(ix_start, new_elements, 0, new_elements.size());
            }

            /*
                At the end of this vector, add all elements in the given vector.
            */
            inline void insert(vector& new_elements) {
                return this->insert(this->size(), new_elements, 0, new_elements.size());
            }
            inline void insert_local(vector& new_elements) {
                return this->insert_local(this->size(), new_elements, 0, new_elements.size());
            }

            /*
                Erases the element at the given index.
            */
            void erase(size_t i);
            
            /*
                Gets the value at the given index.
            */
            std::any get(size_t i);

            /*
                Gets the value at the given local index.
            */
            std::any get_local(size_t i);

            /*
                Empties the contents of this vector and frees any reserved memory.
            */
            void clear();

            /*
                Copies the vector to the host and prints it.
            */
            void print();
            
            size_t size();

            inline size_t local_size() { return this->filled_size; }

            inline bool empty() { return this->data_ptr == nullptr || this->size() == 0; }

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
            void resize_local(size_t N);

            void shrink_to_fit();

            /*
                Sets the stream of this vector.  Affects the maelstrom execution policy.
                The meaning of a stream differs depending on the storage (device, host, distributed, etc.)
                of this vector.
            */
            inline void set_stream(std::any stream) {
                this->stream = stream;
            }

            inline std::any get_stream() {
                return this->stream;
            }

            /*
                Restores the default stream.
            */
            inline void clear_stream() {
                this->stream = get_default_stream(this->mem_type);
            }

            /*
                Elementwise sum of two vectors
            */
            vector operator+(vector& other);

            /*
                Elementwise subtraction of two vectors
            */
            vector operator-(vector& other);

            /*
                Elementwise product of two vectors
            */
            vector operator*(vector& other);

            /*
                Elementwise division of two vectors
            */
            vector operator/(vector& other);

            /*
                Creates a copy of this vector with all elements transformed to the given dtype.
            */
            vector astype(maelstrom::dtype_t new_type);

            /*
                Creates a copy of this vector with the given memory type.
            */
            vector to(maelstrom::storage mem_type);
            

    };

    /*
        If the current vector is already a host vector, returns a non-owning view of the current vector (no data is copied).
        If it is not a host vector, it creates a copy of the vector on the host and returns it.
    */
    maelstrom::vector as_host_vector(maelstrom::vector& vec);

    /*
        If the current vector is already a device vector, returns a non-owning view of the current vector (no data is copied).
        If it is not a device vector, it creates a copy of the vector on the device and returns it.
    */
    maelstrom::vector as_device_vector(maelstrom::vector& vec);

    /*
        Returns a view if view=true, or a copy if view=false, of the current vector with the corresponding primitive type of the vector
    */
    maelstrom::vector as_primitive_vector(maelstrom::vector& vec, bool view=true);

   /*
       Makes a new emtpy vector with the same memory type and data type as the given vector.
        
   */
   inline maelstrom::vector make_vector_like(maelstrom::vector& vec) {
        return maelstrom::vector(
            vec.get_mem_type(),
            vec.get_dtype()
        );
   }

   /*
        Returns a view of the local partition of the given distributed vector.
   */
   inline maelstrom::vector local_view_of(maelstrom::vector& vec) {
        return maelstrom::vector(
            maelstrom::single_storage_of(vec.get_mem_type()),
            vec.get_dtype(),
            vec.data(),
            vec.local_size(),
            true
        );
    }

    /*
        Converts the vector to a distributed vector of the given memory type
        and returns it.
    */
    maelstrom::vector to_dist_vector(maelstrom::vector vec);

   /*
        Makes a new vector of appropriate data type from the given vector of anys
        with the given storage type.
   */
   maelstrom::vector make_vector_from_anys(maelstrom::storage mem_type, std::vector<std::any>& anys);

   /*
        Makes a new vector of the given data type from the given vector of anys
        with the given storage type.
   */
   maelstrom::vector make_vector_from_anys(maelstrom::storage mem_type, maelstrom::dtype_t dtype, std::vector<std::any>& anys);

}