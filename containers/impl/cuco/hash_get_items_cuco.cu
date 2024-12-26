#include "maelstrom/containers/hash_table.h"
#include "maelstrom/util/cuco_utils.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

#include <cuco/dynamic_map.cuh>
#include <cuda/std/atomic>

using cuda::thread_scope_device;

namespace maelstrom {

    template<typename A, typename K, typename V>
    std::pair<std::optional<maelstrom::vector>, std::optional<maelstrom::vector>> cuco_get_hash_table_items(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, bool return_keys, bool return_values) {
        using map_type = cuco::dynamic_map<K, V, thread_scope_device, A>;
        map_type* map = static_cast<map_type*>(data);

        std::optional<maelstrom::vector> keys = return_keys ? std::make_optional(maelstrom::vector(maelstrom::DEVICE, key_dtype, map->get_size())) : std::nullopt;
        std::optional<maelstrom::vector> values = return_values ? std::make_optional(maelstrom::vector(maelstrom::DEVICE, val_dtype, map->get_size())) : std::nullopt;

        size_t offset = 0;
        for(auto& sm : map->view_submaps()) {
            if(return_keys) {
                if(return_values) {
                    sm->retrieve_all(
                        maelstrom::device_tptr_cast<K>(keys->data()) + offset,
                        maelstrom::device_tptr_cast<V>(values->data()) + offset
                    );
                } else {
                    sm->retrieve_all(
                        maelstrom::device_tptr_cast<K>(keys->data()) + offset,
                        thrust::make_discard_iterator()
                    );
                }
            } else {
                if(return_values) {
                    sm->retrieve_all(
                        thrust::make_discard_iterator(),
                        maelstrom::device_tptr_cast<V>(values->data()) + offset
                    );
                }
            }

            offset += sm->get_size();
        }

        return std::make_pair(
            std::move(keys),
            std::move(values)
        );
    }

    template <typename A, typename K>
    std::pair<std::optional<maelstrom::vector>, std::optional<maelstrom::vector>> get_hash_table_items_device_dispatch_val(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, bool return_keys, bool return_values) {
        switch(val_dtype.prim_type) {
            case UINT64:
                return cuco_get_hash_table_items<A, K, uint64_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case UINT32:
                return cuco_get_hash_table_items<A, K, uint32_t>(data, key_dtype, val_dtype, return_keys,  return_values);
            case INT64:
                return cuco_get_hash_table_items<A, K, int64_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case INT32:
                return cuco_get_hash_table_items<A, K, int32_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case FLOAT64:
                return cuco_get_hash_table_items<A, K, double>(data, key_dtype, val_dtype, return_keys, return_values);
            case FLOAT32:
                return cuco_get_hash_table_items<A, K, float>(data, key_dtype, val_dtype, return_keys, return_values);
            case UINT8:
            case INT8:
                throw std::invalid_argument(
                    "single-byte values are not supported in device/managed hash tables, "
                    "consider using a host hash table instead."
                );
        }

        throw std::runtime_error("invalid value type for hash table get items");
    }

    template<typename A>
    std::pair<std::optional<maelstrom::vector>, std::optional<maelstrom::vector>> get_hash_table_items_device_dispatch_key(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, bool return_keys, bool return_values) {
        switch(key_dtype.prim_type) {
            case UINT64:
                return get_hash_table_items_device_dispatch_val<A, uint64_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case UINT32:
                return get_hash_table_items_device_dispatch_val<A, uint32_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case INT64:
                return get_hash_table_items_device_dispatch_val<A, int64_t>(data, key_dtype, val_dtype, return_keys, return_values);
            case INT32:
                return get_hash_table_items_device_dispatch_val<A, int32_t>(data, key_dtype, val_dtype, return_keys, return_values);
        }

        throw std::runtime_error("invalid key type for hash table get items");
    }

    template<>
    std::pair<std::optional<maelstrom::vector>, std::optional<maelstrom::vector>> get_hash_table_items<DEVICE>(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, bool return_keys, bool return_values) {
        return get_hash_table_items_device_dispatch_key<device_allocator_t>(data, key_dtype, val_dtype, return_keys, return_values);
    }

    template<>
    std::pair<std::optional<maelstrom::vector>, std::optional<maelstrom::vector>> get_hash_table_items<MANAGED>(void* data, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, bool return_keys, bool return_values) {
        return get_hash_table_items_device_dispatch_key<managed_allocator_t>(data, key_dtype, val_dtype, return_keys, return_values);
    }

}