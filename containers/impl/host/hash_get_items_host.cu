#include "maelstrom/containers/hash_table.h"
#include "maelstrom/util/host_utils.cuh"
#include <unordered_map>

namespace maelstrom {

    template <typename K, typename V>
    std::pair<std::optional<maelstrom::vector>, std::optional<maelstrom::vector>> get_hash_table_items_host(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, bool return_keys, bool return_values) {
        using map_type = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, maelstrom::maelstrom_host_allocator<std::pair<const K, V>>>;
        map_type* map = static_cast<map_type*>(data);

        std::optional<maelstrom::vector> keys = return_keys ? std::make_optional(maelstrom::vector(maelstrom::HOST, key_dtype, map->size())) : std::nullopt;
        std::optional<maelstrom::vector> values = return_values ? std::make_optional(maelstrom::vector(maelstrom::HOST, val_dtype, map->size())) : std::nullopt;

        size_t k = 0;
        for(auto it = map->begin(); it != map->end(); ++it) {
            if(return_keys) {
                K* kp = static_cast<K*>(keys->data());
                kp[k] = it->first;
            }
            if(return_values) {
                V* vp = static_cast<V*>(values->data());
                vp[k] = it->second;
            }
            ++k;
        }
    
        return std::make_pair(
            keys,
            values
        );
    }

    template<typename K>
    std::pair<std::optional<maelstrom::vector>, std::optional<maelstrom::vector>> get_hash_table_items_host_dispatch_val(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, bool return_keys, bool return_values) {
        switch(val_dtype.prim_type) {
            case UINT64:
                return get_hash_table_items_host<K, uint64_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case UINT32:
                return get_hash_table_items_host<K, uint32_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case UINT8:
                return get_hash_table_items_host<K, uint8_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case INT64:
                return get_hash_table_items_host<K, int64_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case INT32:
                return get_hash_table_items_host<K, int32_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case INT8:
                return get_hash_table_items_host<K, int8_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case FLOAT64:
                return get_hash_table_items_host<K, double>(data, key_dtype, val_dtype, return_keys, return_values);
            case FLOAT32:
                return get_hash_table_items_host<K, float>(data, key_dtype, val_dtype, return_keys, return_values);
        }

        throw std::runtime_error("invalid value type for hash table get items");
    }

    std::pair<std::optional<maelstrom::vector>, std::optional<maelstrom::vector>> get_hash_table_items_host_dispatch_key(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, bool return_keys, bool return_values) {
        switch(key_dtype.prim_type) {
            case UINT64:
                return get_hash_table_items_host_dispatch_val<uint64_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case UINT32:
                return get_hash_table_items_host_dispatch_val<uint32_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case INT64:
                return get_hash_table_items_host_dispatch_val<int64_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case INT32:
                return get_hash_table_items_host_dispatch_val<int32_t>(data, key_dtype, val_dtype, return_keys, return_values);
        }

        throw std::runtime_error("invalid key type for hash table get items");
    }

    template<>
    std::pair<std::optional<maelstrom::vector>, std::optional<maelstrom::vector>> get_hash_table_items<HOST>(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, bool return_keys, bool return_values) {
        return get_hash_table_items_host_dispatch_key(data, key_dtype, val_dtype, return_keys, return_values);
    }

}