#include "maelstrom/containers/hash_table.h"

#include "maelstrom/util/cuco_utils.cuh"
#include <cuco/dynamic_map.cuh>
#include <cuda/std/atomic>

using cuda::thread_scope_device;

namespace maelstrom {

    template<typename A, typename K, typename V>
    size_t cuco_get_hash_table_size(void* data) {
        using map_type = cuco::dynamic_map<K, V, thread_scope_device, A>;
        return static_cast<map_type*>(data)->get_size();
    }

    template<typename A, typename K>
    size_t size_hash_table_device_dispatch_val(void* data, maelstrom::dtype_t val_dtype) {
        switch(val_dtype.prim_type) {
            case UINT64:
                return cuco_get_hash_table_size<A, K, uint64_t>(data);
            case UINT32:
                return cuco_get_hash_table_size<A, K, uint32_t>(data);
            case INT64:
                return cuco_get_hash_table_size<A, K, int64_t>(data);
            case INT32:
                return cuco_get_hash_table_size<A, K, int32_t>(data);
            case FLOAT64:
                return cuco_get_hash_table_size<A, K, double>(data);
            case FLOAT32:
                return cuco_get_hash_table_size<A, K, float>(data);
            case UINT8:
            case INT8:
                throw std::invalid_argument(
                    "single-byte values are not supported in device/managed hash tables, "
                    "consider using a host hash table instead."
                );
        }

        throw std::invalid_argument("invalid value dtype for hash table size");
    }

    template<typename A>
    size_t size_hash_table_device_dispatch_key(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype) {
        switch(key_dtype.prim_type) {
            case UINT64:
                return size_hash_table_device_dispatch_val<A, uint64_t>(data, val_dtype);
            case UINT32:
                return size_hash_table_device_dispatch_val<A, uint32_t>(data, val_dtype);
            case INT64:
                return size_hash_table_device_dispatch_val<A, int64_t>(data, val_dtype);
            case INT32:
                return size_hash_table_device_dispatch_val<A, int32_t>(data, val_dtype);
        }

        throw std::invalid_argument("invalid key dtype for hash table size");
    }

    template<>
    size_t size_hash_table<DEVICE>(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype) {
        return size_hash_table_device_dispatch_key<device_allocator_t>(data, key_dtype, val_dtype);
    }

    template<>
    size_t size_hash_table<MANAGED>(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype) {
        return size_hash_table_device_dispatch_key<managed_allocator_t>(data, key_dtype, val_dtype);
    }

}