#include "maelstrom/containers/hash_table.h"
#include "maelstrom/storage/allocators.cuh"
#include "maelstrom/util/cuco_utils.cuh"

#include <cuco/dynamic_map.cuh>
#include <cuda/std/atomic>

using cuda::thread_scope_device;

namespace maelstrom {
    template<typename A, typename K, typename V>
    void* cuco_instantiate_hash_table_device(size_t initial_size) {
        cuco::dynamic_map<K, V, thread_scope_device, A>* data = new cuco::dynamic_map<K, V, thread_scope_device, A>{
            initial_size,
            cuco::empty_key<K>(std::numeric_limits<K>::max()),
            cuco::empty_value<V>(std::numeric_limits<V>::max()),
            cuco::erased_key<K>(std::numeric_limits<K>::max() - 1),
            A()
        };

        return static_cast<void*>(data);
    }

    template<typename A, typename K>
    void* instantiate_hash_table_device_dispatch_val(maelstrom::dtype_t val_dtype, size_t initial_size) {
        switch(val_dtype.prim_type) {
            case UINT64:
                return cuco_instantiate_hash_table_device<A, K, uint64_t>(initial_size);
            case UINT32:
                return cuco_instantiate_hash_table_device<A, K, uint32_t>(initial_size);
            case INT64:
                return cuco_instantiate_hash_table_device<A, K, int64_t>(initial_size);
            case INT32:
                return cuco_instantiate_hash_table_device<A, K, int32_t>(initial_size);
            case FLOAT64:
                return cuco_instantiate_hash_table_device<A, K, double>(initial_size);
            case FLOAT32:
                return cuco_instantiate_hash_table_device<A, K, float>(initial_size);
            case UINT8:
            case INT8:
                throw std::invalid_argument(
                    "single-byte values are not supported in device/managed hash tables, "
                    "consider using a host hash table instead."
                );
        }

        throw std::runtime_error("illegal value type for device hash table");
    }

    template <typename A>
    void* instantiate_hash_table_device_dispatch_key(maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, size_t initial_size) {
        switch(key_dtype.prim_type) {
            case UINT64:
                return instantiate_hash_table_device_dispatch_val<A, uint64_t>(val_dtype, initial_size);
            case UINT32:
                return instantiate_hash_table_device_dispatch_val<A, uint32_t>(val_dtype, initial_size);
            case INT64:
                return instantiate_hash_table_device_dispatch_val<A, int64_t>(val_dtype, initial_size);
            case INT32:
                return instantiate_hash_table_device_dispatch_val<A, int32_t>(val_dtype, initial_size);
        }

        throw std::runtime_error("Invalid key dtype for device hash table");
    }

    template<>
    void* instantiate_hash_table<DEVICE>(maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, size_t initial_size) {
        return instantiate_hash_table_device_dispatch_key<device_allocator_t>(key_dtype, val_dtype, initial_size);
    }

    template<>
    void* instantiate_hash_table<MANAGED>(maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, size_t initial_size){ 
        return instantiate_hash_table_device_dispatch_key<managed_allocator_t>(key_dtype, val_dtype, initial_size);
    }

}