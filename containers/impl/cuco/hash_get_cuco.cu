#include "maelstrom/containers/hash_table.h"
#include "maelstrom/util/cuco_utils.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

#include <cuco/dynamic_map.cuh>
#include <cuda/std/atomic>

using cuda::thread_scope_device;

namespace maelstrom {
    template<typename A, typename K, typename V>
    maelstrom::vector cuco_get_hash_table(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_type, bool return_values) {
        using map_type = cuco::dynamic_map<K, V, thread_scope_device, A>;
        map_type* map = static_cast<map_type*>(data);

        if(return_values) {
            maelstrom::vector found(
                keys.get_mem_type(),
                value_type,
                keys.size()
            );

            map->find(
                maelstrom::device_tptr_cast<K>(keys.data()),
                maelstrom::device_tptr_cast<K>(keys.data()) + keys.size(),
                maelstrom::device_tptr_cast<V>(found.data())
            );

            return found;
        }

        // will return booleans instead
        maelstrom::vector found(
            keys.get_mem_type(),
            uint8,
            keys.size()
        );

        map->contains(
            maelstrom::device_tptr_cast<K>(keys.data()),
            maelstrom::device_tptr_cast<K>(keys.data()) + keys.size(),
            maelstrom::device_tptr_cast<uint8_t>(found.data())
        );

        return found;
    }

    template<typename A, typename K>
    maelstrom::vector get_hash_table_device_dispatch_val(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_type, bool return_values) {
        switch(value_type.prim_type) {
            case UINT64:
                return cuco_get_hash_table<A, K, uint64_t>(data, keys, value_type, return_values);
            case UINT32:
                return cuco_get_hash_table<A, K, uint32_t>(data, keys, value_type, return_values);
            case UINT8:
                return cuco_get_hash_table<A, K, uint8_t>(data, keys, value_type, return_values);
            case INT64:
                return cuco_get_hash_table<A, K, int64_t>(data, keys, value_type, return_values);
            case INT32:
                return cuco_get_hash_table<A, K, int32_t>(data, keys, value_type, return_values);
            case INT8:
                return cuco_get_hash_table<A, K, int8_t>(data, keys, value_type, return_values);
            case FLOAT64:
                return cuco_get_hash_table<A, K, double>(data, keys, value_type, return_values);
            case FLOAT32:
                return cuco_get_hash_table<A, K, float>(data, keys, value_type, return_values);
        }

        throw std::runtime_error("invalid value type for hash table (insert)");
    }

    template<typename A>
    maelstrom::vector get_hash_table_device_dispatch_key(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_type, bool return_values) {
        switch(keys.get_dtype().prim_type) {
            case UINT64:
                return get_hash_table_device_dispatch_val<A, uint64_t>(data, keys, value_type, return_values);
            case UINT32:
                return get_hash_table_device_dispatch_val<A, uint32_t>(data, keys, value_type, return_values);
            case INT64:
                return get_hash_table_device_dispatch_val<A, int64_t>(data, keys, value_type, return_values);
            case INT32:
                return get_hash_table_device_dispatch_val<A, int32_t>(data, keys, value_type, return_values);
        }

        throw std::runtime_error("invalid key type for hash table (insert)");
    }

    template<>
    maelstrom::vector get_hash_table<DEVICE>(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_type, bool return_values) {
        return get_hash_table_device_dispatch_key<device_allocator_t>(data, keys, value_type, return_values);
    }

    template<>
    maelstrom::vector get_hash_table<MANAGED>(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_type, bool return_values) {
        return get_hash_table_device_dispatch_key<managed_allocator_t>(data, keys, value_type, return_values);
    }

}