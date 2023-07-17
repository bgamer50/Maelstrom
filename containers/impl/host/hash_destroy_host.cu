#include "maelstrom/containers/hash_table.h"
#include "maelstrom/util/host_utils.cuh"
#include <unordered_map>

namespace maelstrom {

    template<typename K, typename V>
    void h_destroy_hash_table(void* data) {
        using h_alloc_t = maelstrom::maelstrom_host_allocator<std::pair<const K, V>>;
        delete static_cast<std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, h_alloc_t>*>(data);
    }

    template<typename K>
    void destroy_hash_table_host_dispatch_val(void* data, maelstrom::dtype_t val_dtype) {
        switch(val_dtype.prim_type) {
            case UINT64:
                return h_destroy_hash_table<K, uint64_t>(data);
            case UINT32:
                return h_destroy_hash_table<K, uint32_t>(data);
            case UINT8:
                return h_destroy_hash_table<K, uint8_t>(data);
            case INT64:
                return h_destroy_hash_table<K, int64_t>(data);
            case INT32:
                return h_destroy_hash_table<K, int32_t>(data);
            case INT8:
                return h_destroy_hash_table<K, int8_t>(data);
            case FLOAT64:
                return h_destroy_hash_table<K, double>(data);
            case FLOAT32:
                return h_destroy_hash_table<K, float>(data);
        }

        throw std::runtime_error("invalid value type for hash table (construct)");
    }

    void destroy_hash_table_host_dispatch_key(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype) {
        switch(key_dtype.prim_type) {
            case UINT64:
                return destroy_hash_table_host_dispatch_val<uint64_t>(data, val_dtype);
            case UINT32:
                return destroy_hash_table_host_dispatch_val<uint32_t>(data, val_dtype);
            case INT64:
                return destroy_hash_table_host_dispatch_val<int64_t>(data, val_dtype);
            case INT32:
                return destroy_hash_table_host_dispatch_val<int32_t>(data, val_dtype);
        }

        throw std::runtime_error("invalid key type for hash table (construct)");
    }

    template<>
    void destroy_hash_table<HOST>(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype){ 
        return destroy_hash_table_host_dispatch_key(data, key_dtype, val_dtype);
    }

}