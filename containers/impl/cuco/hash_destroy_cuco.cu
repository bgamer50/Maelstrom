#include "maelstrom/containers/hash_table.h"
#include "maelstrom/storage/allocators.cuh"
#include "maelstrom/util/cuco_utils.cuh"

#include <cuco/dynamic_map.cuh>
#include <cuda/std/atomic>

using cuda::thread_scope_device;

namespace maelstrom {
    template<typename A, typename K, typename V>
    void cuco_destroy_hash_table(void* data) {
        delete static_cast<cuco::dynamic_map<K, V, thread_scope_device, A>*>(data);
    }

    template<typename A, typename K>
    void destroy_hash_table_device_dispatch_val(void* data, maelstrom::dtype_t val_dtype) {
        switch(val_dtype.prim_type) {
            case UINT64:
                return cuco_destroy_hash_table<A, K, uint64_t>(data);
            case UINT32:
                return cuco_destroy_hash_table<A, K, uint32_t>(data);
            case INT64:
                return cuco_destroy_hash_table<A, K, int64_t>(data);
            case INT32:
                return cuco_destroy_hash_table<A, K, int32_t>(data);
            case FLOAT64:
                return cuco_destroy_hash_table<A, K, double>(data);
            case FLOAT32:
                return cuco_destroy_hash_table<A, K, float>(data);
            case UINT8:
            case INT8:
                throw std::invalid_argument(
                    "single-byte values are not supported in device/managed hash tables, "
                    "consider using a host hash table instead."
                );
        }

        throw std::runtime_error("invalid dtype for hash table (destroy)");
    }

    template<typename A>
    void destroy_hash_table_device_dispatch_key(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype) {
        switch(key_dtype.prim_type) {
            case UINT64:
                return destroy_hash_table_device_dispatch_val<A, uint64_t>(data, val_dtype);
            case UINT32:
                return destroy_hash_table_device_dispatch_val<A, uint32_t>(data, val_dtype);
            case INT64:
                return destroy_hash_table_device_dispatch_val<A, int64_t>(data, val_dtype);
            case INT32:
                return destroy_hash_table_device_dispatch_val<A, int32_t>(data, val_dtype);
        }

        throw std::runtime_error("Invalid key type for device hash table (destroy)");
    }

    template<>
    void destroy_hash_table<DEVICE>(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype) {
        return destroy_hash_table_device_dispatch_key<device_allocator_t>(data, key_dtype, val_dtype);
    }

    template<>
    void destroy_hash_table<MANAGED>(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype) {
        return destroy_hash_table_device_dispatch_key<managed_allocator_t>(data, key_dtype, val_dtype);
    }
}