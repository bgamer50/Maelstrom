#include "maelstrom/containers/hash_table.h"
#include "maelstrom/util/host_utils.cuh"

#include <unordered_map>

namespace maelstrom {
    template<typename K, typename V>
    void h_insert_hash_table(void* data, maelstrom::vector& keys, maelstrom::vector& values) {
        using map_type = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, maelstrom::maelstrom_host_allocator<std::pair<const K, V>>>;
        map_type* map = static_cast<map_type*>(data);

        for(size_t k = 0; k < keys.size(); ++k) {
            map->insert_or_assign(
                static_cast<K*>(keys.data())[k],
                static_cast<V*>(values.data())[k]   
            );
        }
    }

    template<typename K>
    void insert_hash_table_host_dispatch_val(void* data, maelstrom::vector& keys, maelstrom::vector& values) {
        switch(values.get_dtype().prim_type) {
            case UINT64:
                return h_insert_hash_table<K, uint64_t>(data, keys, values);
            case UINT32:
                return h_insert_hash_table<K, uint32_t>(data, keys, values);
            case UINT8:
                return h_insert_hash_table<K, uint8_t>(data, keys, values);
            case INT64:
                return h_insert_hash_table<K, int64_t>(data, keys, values);
            case INT32:
                return h_insert_hash_table<K, int32_t>(data, keys, values);
            case INT8:
                return h_insert_hash_table<K, int8_t>(data, keys, values);
            case FLOAT64:
                return h_insert_hash_table<K, double>(data, keys, values);
            case FLOAT32:
                return h_insert_hash_table<K, float>(data, keys, values);
        }

        throw std::runtime_error("invalid value type for hash table (insert)");
    }

    void insert_hash_table_host_dispatch_key(void* data, maelstrom::vector& keys, maelstrom::vector& values) {
        switch(keys.get_dtype().prim_type) {
            case UINT64:
                return insert_hash_table_host_dispatch_val<uint64_t>(data, keys, values);
            case UINT32:
                return insert_hash_table_host_dispatch_val<uint32_t>(data, keys, values);
            case INT64:
                return insert_hash_table_host_dispatch_val<int64_t>(data, keys, values);
            case INT32:
                return insert_hash_table_host_dispatch_val<int32_t>(data, keys, values);
        }

        throw std::runtime_error("invalid key type for hash table (insert)");
    }

    template<>
    void insert_hash_table<HOST>(void* data, maelstrom::vector& keys, maelstrom::vector& values) {
        return insert_hash_table_host_dispatch_key(data, keys, values);
    }

}