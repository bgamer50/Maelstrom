#include "maelstrom/containers/hash_table.h"
#include "maelstrom/util/host_utils.cuh"
#include <unordered_map>

namespace maelstrom {

    template<typename K, typename V>
    void* h_instantiate_hash_table(size_t initial_size) {
        using h_alloc_t = maelstrom::maelstrom_host_allocator<std::pair<const K, V>>;
        std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, h_alloc_t>* data = new std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, h_alloc_t>();
        data->reserve(initial_size);
        return static_cast<void*>(data);
    }

    template<typename K>
    void* instantiate_hash_table_host_dispatch_val(maelstrom::dtype_t val_dtype, size_t initial_size) {
        switch(val_dtype.prim_type) {
            case UINT64:
                return h_instantiate_hash_table<K, uint64_t>(initial_size);
            case UINT32:
                return h_instantiate_hash_table<K, uint32_t>(initial_size);
            case UINT8:
                return h_instantiate_hash_table<K, uint8_t>(initial_size);
            case INT64:
                return h_instantiate_hash_table<K, int64_t>(initial_size);
            case INT32:
                return h_instantiate_hash_table<K, int32_t>(initial_size);
            case INT8:
                return h_instantiate_hash_table<K, int8_t>(initial_size);
            case FLOAT64:
                return h_instantiate_hash_table<K, double>(initial_size);
            case FLOAT32:
                return h_instantiate_hash_table<K, float>(initial_size);
        }

        throw std::runtime_error("invalid value type for hash table (construct)");
    }

    void* instantiate_hash_table_host_dispatch_key(maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, size_t initial_size) {
        switch(key_dtype.prim_type) {
            case UINT64:
                return instantiate_hash_table_host_dispatch_val<uint64_t>(val_dtype, initial_size);
            case UINT32:
                return instantiate_hash_table_host_dispatch_val<uint32_t>(val_dtype, initial_size);
            case INT64:
                return instantiate_hash_table_host_dispatch_val<int64_t>(val_dtype, initial_size);
            case INT32:
                return instantiate_hash_table_host_dispatch_val<int32_t>(val_dtype, initial_size);
        }

        throw std::runtime_error("invalid key type for hash table (construct)");
    }

    template<>
    void* instantiate_hash_table<HOST>(maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, size_t initial_size){ 
        return instantiate_hash_table_host_dispatch_key(key_dtype, val_dtype, initial_size);
    }

}