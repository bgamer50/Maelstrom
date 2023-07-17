#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {

    /*
        Instantiates a new hash table.  Returns a pointer to the data corresponding to the
        new hash table.
    */
    template<maelstrom::storage S>
    void* instantiate_hash_table(maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, size_t initial_size);

    /*
        Destroys the hash table pointed to by the given data pointer.
    */
    template<maelstrom::storage S>
    void destroy_hash_table(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype);

    /*
        Inserts the key-value pairs into the hash table.
    */
    template<maelstrom::storage S>
    void insert_hash_table(void* data, maelstrom::vector& keys, maelstrom::vector& values);

    /*
        If return_values=true, returns the values for the given keys
        (returns the max value for the value type if its corresponding key was not found).
        If return_values=false, returns booleans for whether or not the key was found.
    */
    template<maelstrom::storage S>
    maelstrom::vector get_hash_table(void* data, maelstrom::vector& keys, maelstrom::dtype_t val_dtype, bool return_values=true);

    /*
        Removes the given keys and their values from the given hash table.
    */
    template<maelstrom::storage S>
    void remove_hash_table(void* data, maelstrom::vector& keys, maelstrom::dtype_t val_dtype);

    class hash_table {
        private:
            void* data_ptr;

            maelstrom::storage mem_type;
            maelstrom::dtype_t key_dtype;
            maelstrom::dtype_t val_dtype;

        public:
            hash_table(maelstrom::storage mem_type, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, size_t initial_size=62);

            ~hash_table() noexcept(false);

            void set(maelstrom::vector& keys, maelstrom::vector& vals);

            maelstrom::vector get(maelstrom::vector& keys);

            void remove(maelstrom::vector& keys);

            maelstrom::vector contains(maelstrom::vector& keys);

    };

}