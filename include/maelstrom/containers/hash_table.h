#pragma once

#include "maelstrom/containers/vector.h"
#include <optional>

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
        Returns the items in this has table.
    */
    template<maelstrom::storage S>
    std::pair<std::optional<maelstrom::vector>, std::optional<maelstrom::vector>> get_hash_table_items(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, bool return_keys, bool return_values);

    /*
        Removes the given keys and their values from the given hash table.
    */
    template<maelstrom::storage S>
    void remove_hash_table(void* data, maelstrom::vector& keys, maelstrom::dtype_t val_dtype);

    template<maelstrom::storage S>
    size_t size_hash_table(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype);

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

            maelstrom::vector get_keys();

            maelstrom::vector get_values();

            std::pair<maelstrom::vector, maelstrom::vector> get_items();

            size_t size();

            inline std::any key_not_found() { return maelstrom::max_value(this->key_dtype); }

            inline std::any val_not_found() { return maelstrom::max_value(this->val_dtype); }

            inline maelstrom::storage get_mem_type() { return this->mem_type; }

            inline maelstrom::dtype_t get_key_dtype() { return this->key_dtype; }

            inline maelstrom::dtype_t get_val_dtype() { return this->val_dtype; }

    };

}