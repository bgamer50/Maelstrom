#include "maelstrom/containers/hash_table.h"
#include "maelstrom/util/cuco_utils.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

#include <cuco/dynamic_map.cuh>
#include <cuda/std/atomic>

using cuda::thread_scope_device;

namespace maelstrom {
    template<typename A, typename K, typename V>
    void cuco_insert_hash_table(void* data, maelstrom::vector& keys, maelstrom::vector& values) {
        using map_type = cuco::dynamic_map<K, V, thread_scope_device, A>;
        map_type* map = static_cast<map_type*>(data);

        auto pairs_begin = thrust::make_zip_iterator(
            maelstrom::device_tptr_cast<K>(keys.data()),
            maelstrom::device_tptr_cast<V>(values.data())
        );

        map->insert(
            pairs_begin,
            pairs_begin + keys.size()
        );
    }

    template<typename A, typename K>
    void insert_hash_table_device_dispatch_val(void* data, maelstrom::vector& keys, maelstrom::vector& values) {
        switch(values.get_dtype().prim_type) {
            case UINT64:
                return cuco_insert_hash_table<A, K, uint64_t>(data, keys, values);
            case UINT32:
                return cuco_insert_hash_table<A, K, uint32_t>(data, keys, values);
            case UINT8:
                return cuco_insert_hash_table<A, K, uint8_t>(data, keys, values);
            case INT64:
                return cuco_insert_hash_table<A, K, int64_t>(data, keys, values);
            case INT32:
                return cuco_insert_hash_table<A, K, int32_t>(data, keys, values);
            case INT8:
                return cuco_insert_hash_table<A, K, int8_t>(data, keys, values);
            case FLOAT64:
                return cuco_insert_hash_table<A, K, double>(data, keys, values);
            case FLOAT32:
                return cuco_insert_hash_table<A, K, float>(data, keys, values);
        }

        throw std::runtime_error("invalid value type for hash table (insert)");
    }

    template<typename A>
    void insert_hash_table_device_dispatch_key(void* data, maelstrom::vector& keys, maelstrom::vector& values) {
        switch(keys.get_dtype().prim_type) {
            case UINT64:
                return insert_hash_table_device_dispatch_val<A, uint64_t>(data, keys, values);
            case UINT32:
                return insert_hash_table_device_dispatch_val<A, uint32_t>(data, keys, values);
            case INT64:
                return insert_hash_table_device_dispatch_val<A, int64_t>(data, keys, values);
            case INT32:
                return insert_hash_table_device_dispatch_val<A, int32_t>(data, keys, values);
        }

        throw std::runtime_error("invalid key type for hash table (insert)");
    }

    template<>
    void insert_hash_table<DEVICE>(void* data, maelstrom::vector& keys, maelstrom::vector& values) {
        return insert_hash_table_device_dispatch_key<device_allocator_t>(data, keys, values);
    }

    template<>
    void insert_hash_table<MANAGED>(void* data, maelstrom::vector& keys, maelstrom::vector& values) {
        return insert_hash_table_device_dispatch_key<managed_allocator_t>(data, keys, values);
    }

}