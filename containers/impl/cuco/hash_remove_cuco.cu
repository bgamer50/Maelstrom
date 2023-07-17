#include "maelstrom/containers/hash_table.h"
#include "maelstrom/util/cuco_utils.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

#include <cuco/dynamic_map.cuh>
#include <cuda/std/atomic>

using cuda::thread_scope_device;

namespace maelstrom {
    template<typename A, typename K, typename V>
    void cuco_remove_hash_table(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_dtype) {
        using map_type = cuco::dynamic_map<K, V, thread_scope_device, A>;
        map_type* map = static_cast<map_type*>(data);

        map->erase(
            maelstrom::device_tptr_cast<K>(keys.data()),
            maelstrom::device_tptr_cast<K>(keys.data()) + keys.size()
        );
    }

    template<typename A, typename K>
    void remove_hash_table_device_dispatch_val(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_dtype) {
        switch(value_dtype.prim_type) {
            case UINT64:
                return cuco_remove_hash_table<A, K, uint64_t>(data, keys, value_dtype);
            case UINT32:
                return cuco_remove_hash_table<A, K, uint32_t>(data, keys, value_dtype);
            case INT64:
                return cuco_remove_hash_table<A, K, int64_t>(data, keys, value_dtype);
            case INT32:
                return cuco_remove_hash_table<A, K, int32_t>(data, keys, value_dtype);
            case FLOAT64:
                return cuco_remove_hash_table<A, K, double>(data, keys, value_dtype);
            case FLOAT32:
                return cuco_remove_hash_table<A, K, float>(data, keys, value_dtype);
            case UINT8:
            case INT8:
                throw std::runtime_error("single-byte values are currently immutable in device hash tables!");
        }

        throw std::runtime_error("invalid value type for hash table (insert)");
    }

    template<typename A>
    void remove_hash_table_device_dispatch_key(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_dtype) {
        switch(keys.get_dtype().prim_type) {
            case UINT64:
                return remove_hash_table_device_dispatch_val<A, uint64_t>(data, keys, value_dtype);
            case UINT32:
                return remove_hash_table_device_dispatch_val<A, uint32_t>(data, keys, value_dtype);
            case INT64:
                return remove_hash_table_device_dispatch_val<A, int64_t>(data, keys, value_dtype);
            case INT32:
                return remove_hash_table_device_dispatch_val<A, int32_t>(data, keys, value_dtype);
        }

        throw std::runtime_error("invalid key type for hash table (insert)");
    }

    template<>
    void remove_hash_table<DEVICE>(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_dtype) {
        return remove_hash_table_device_dispatch_key<device_allocator_t>(data, keys, value_dtype);
    }

    template<>
    void remove_hash_table<MANAGED>(void* data, maelstrom::vector& keys, maelstrom::dtype_t value_dtype) {
        return remove_hash_table_device_dispatch_key<managed_allocator_t>(data, keys, value_dtype);
    }

}