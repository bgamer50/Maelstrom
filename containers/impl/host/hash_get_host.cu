#include "maelstrom/containers/hash_table.h"
#include "maelstrom/util/host_utils.cuh"
#include <limits>

namespace maelstrom {
    template<typename K, typename V>
    maelstrom::vector h_get_hash_table(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_type, bool return_values) {
        using map_type = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, maelstrom::maelstrom_host_allocator<std::pair<const K, V>>>;
        map_type* map = static_cast<map_type*>(data);

        maelstrom::vector found(
            keys.get_mem_type(),
            return_values ? value_type : uint8,
            keys.size()
        );

        for(size_t k = 0; k < keys.size(); ++k) {
            auto it = map->find(
                static_cast<K*>(keys.data())[k]
            );

            if(return_values) {
                if(it == map->end()) static_cast<V*>(found.data())[k] = std::numeric_limits<V>::max();
                else static_cast<V*>(found.data())[k] = it->second;
            } else {
                if(it == map->end()) static_cast<uint8_t*>(found.data())[k] = false;
                else static_cast<uint8_t*>(found.data())[k] = true;
            }
        }

        return found;
    }

    template<typename K>
    maelstrom::vector get_hash_table_host_dispatch_val(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_type, bool return_values) {
        switch(value_type.prim_type) {
            case UINT64:
                return h_get_hash_table<K, uint64_t>(data, keys, value_type, return_values);
            case UINT32:
                return h_get_hash_table<K, uint32_t>(data, keys, value_type, return_values);
            case UINT8:
                return h_get_hash_table<K, uint8_t>(data, keys, value_type, return_values);
            case INT64:
                return h_get_hash_table<K, int64_t>(data, keys, value_type, return_values);
            case INT32:
                return h_get_hash_table<K, int32_t>(data, keys, value_type, return_values);
            case INT8:
                return h_get_hash_table<K, int8_t>(data, keys, value_type, return_values);
            case FLOAT64:
                return h_get_hash_table<K, double>(data, keys, value_type, return_values);
            case FLOAT32:
                return h_get_hash_table<K, float>(data, keys, value_type, return_values);
        }

        throw std::runtime_error("invalid value type for hash table (insert)");
    }

    maelstrom::vector get_hash_table_host_dispatch_key(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_type, bool return_values) {
        switch(keys.get_dtype().prim_type) {
            case UINT64:
                return get_hash_table_host_dispatch_val<uint64_t>(data, keys, value_type, return_values);
            case UINT32:
                return get_hash_table_host_dispatch_val<uint32_t>(data, keys, value_type, return_values);
            case INT64:
                return get_hash_table_host_dispatch_val<int64_t>(data, keys, value_type, return_values);
            case INT32:
                return get_hash_table_host_dispatch_val<int32_t>(data, keys, value_type, return_values);
        }

        throw std::runtime_error("invalid key type for hash table (insert)");
    }

    template<>
    maelstrom::vector get_hash_table<HOST>(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_type, bool return_values) {
        return get_hash_table_host_dispatch_key(data, keys, value_type, return_values);
    }

}