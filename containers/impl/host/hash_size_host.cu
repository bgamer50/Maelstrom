#include "maelstrom/containers/hash_table.h"
#include "maelstrom/util/host_utils.cuh"

namespace maelstrom {

    template<typename K, typename V>
    size_t size_hash_table_host(void*data) {
        using map_type = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, maelstrom::maelstrom_host_allocator<std::pair<const K, V>>>;
        return static_cast<map_type*>(data)->size();
    }

    template <typename K>
    size_t size_hash_table_host_dispatch_val(void* data, maelstrom::dtype_t val_dtype) {
        switch(val_dtype.prim_type) {
            case UINT64:
                return size_hash_table_host<K, uint64_t>(data);
            case UINT32:
                return size_hash_table_host<K, uint32_t>(data);
            case UINT8:
                return size_hash_table_host<K, uint8_t>(data);
            case INT64:
                return size_hash_table_host<K, int64_t>(data);
            case INT32:
                return size_hash_table_host<K, int32_t>(data);
            case INT8:
                return size_hash_table_host<K, int8_t>(data);
            case FLOAT64:
                return size_hash_table_host<K, double>(data);
            case FLOAT32:
                return size_hash_table_host<K, float>(data);
        }

        throw std::runtime_error("invalid value dtype for hash table size");
    }
    
    size_t size_hash_table_host_dispatch_key(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype) {
        switch(key_dtype.prim_type) {
            case UINT64:
                return size_hash_table_host_dispatch_val<uint64_t>(data, val_dtype);
            case UINT32:
                return size_hash_table_host_dispatch_val<uint32_t>(data, val_dtype);
            case INT64:
                return size_hash_table_host_dispatch_val<int64_t>(data, val_dtype);
            case INT32:
                return size_hash_table_host_dispatch_val<int32_t>(data, val_dtype);
        }

        throw std::runtime_error("invalid key dtype for hash table size");
    }

    template<>
    size_t size_hash_table<HOST>(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype) {
        return size_hash_table_host_dispatch_key(data, key_dtype, val_dtype);
    }

}