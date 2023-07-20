#include "maelstrom/containers/vector.h"
#include "maelstrom/util/cuda_utils.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"
#include "maelstrom/thrust_utils/execution.cuh"

namespace maelstrom {
    namespace sparse {

        template <typename I, typename R>
        __global__ void k_query_adjacency_get_mem(I* row, I* col, R* rel, I* query, size_t query_size, R* query_rel, size_t query_rel_size, size_t* output_memory) {
            const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;
            
            const bool filter_rel = query_rel_size > 0;

            for(size_t i = index; i < query_size; i += stride) {
                size_t query_i = query[i];

                size_t start = row[query_i];
                size_t end = row[query_i + 1];

                if(!filter_rel) {
                    output_memory[i] = end - start;
                } else {
                    output_memory[i] = 0;
                    for(size_t j = start; j < end; ++j) {   
                        bool filter_pass = false;
                        R rel_j = rel[j];
                        for(size_t r = 0; r < query_rel_size; ++r) {
                            if(rel_j == query_rel[r]) {
                                filter_pass = true;
                                //break;
                            }
                        }

                        if(filter_pass) {
                            output_memory[i] += 1;
                        }
                    }
                } 
            }    
        }

        template <typename I, typename V, typename R>
        __global__ void k_query_adjacency(I* row, I* col, R* rel, V* val, I* query, size_t query_size, R* query_rel, size_t query_rel_size, size_t* ps, size_t* origin, I* adjacent, R* rel_adjacent, V* val_adjacent, bool return_adj, bool return_rel, bool return_val) {
            const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;
            
            const bool filter_rel = query_rel_size > 0;

            for(size_t i = index; i < query_size; i += stride) {
                size_t query_i = query[i];
                size_t output_index = i==0 ? 0 : ps[i - 1];

                size_t start = row[query_i];
                size_t end = row[query_i + 1];

                for(size_t j = start; j < end; ++j) {
                    bool filter_pass = true;
                    if(filter_rel) {
                        filter_pass = false;
                        R rel_j = rel[j];
                        for(size_t r = 0; r < query_rel_size; ++r) {
                            if(rel_j == query_rel[r]) {
                                filter_pass = true;
                                //break;
                            }
                        }
                    }

                    if(filter_pass) {
                        origin[output_index] = i;
                        if(return_adj) adjacent[output_index] = col[j];
                        if(return_rel) rel_adjacent[output_index] = rel[j];
                        if(return_val) val_adjacent[output_index] = val[j];
                        ++output_index;
                    }
                }
            }
        }

        template <typename E, typename I, typename V, typename R>
        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> exec_query_adjacency_device(E execution_policy,
                                                                                                                    maelstrom::vector& row,
                                                                                                                    maelstrom::vector& col,
                                                                                                                    maelstrom::vector& val,
                                                                                                                    maelstrom::vector& rel,
                                                                                                                    maelstrom::vector& ix,
                                                                                                                    maelstrom::vector& rel_types,
                                                                                                                    bool return_inner,
                                                                                                                    bool return_values,
                                                                                                                    bool return_relations) 
        {
            size_t ix_size = ix.size();

            maelstrom::vector output_memory(
                row.get_mem_type(),
                maelstrom::uint64,
                ix_size
            );

            const size_t num_blocks = maelstrom::cuda::num_blocks(ix_size, MAELSTROM_DEFAULT_BLOCK_SIZE);

            k_query_adjacency_get_mem<<<num_blocks, MAELSTROM_DEFAULT_BLOCK_SIZE>>>(
                static_cast<I*>(row.data()),
                static_cast<I*>(col.data()),
                static_cast<R*>(rel.data()),
                static_cast<I*>(ix.data()),
                ix_size,
                static_cast<R*>(rel_types.data()),
                rel_types.size(),
                static_cast<size_t*>(output_memory.data())
            );
            cudaDeviceSynchronize();
            maelstrom::cuda::cudaCheckErrors("k_query_adjacency_get_mem");

            // in-place prefix sum
            thrust::inclusive_scan(
                execution_policy,
                maelstrom::device_tptr_cast<size_t>(output_memory.data()),
                maelstrom::device_tptr_cast<size_t>(output_memory.data()) + output_memory.size(),
                maelstrom::device_tptr_cast<size_t>(output_memory.data())
            );

            size_t output_size = std::any_cast<size_t>(output_memory.get(output_memory.size() - 1));

            maelstrom::vector origin(row.get_mem_type(), uint64, output_size);
            maelstrom::vector adjacent(row.get_mem_type(), row.get_dtype(), return_inner ? output_size : 0);
            maelstrom::vector rel_adjacent(row.get_mem_type(), rel.get_dtype(), return_relations ? output_size : 0);
            maelstrom::vector val_adjacent(row.get_mem_type(), val.get_dtype(), return_values ? output_size : 0);

            k_query_adjacency<<<num_blocks, MAELSTROM_DEFAULT_BLOCK_SIZE>>>(
                static_cast<I*>(row.data()),
                static_cast<I*>(col.data()),
                static_cast<R*>(rel.data()),
                static_cast<V*>(val.data()),
                static_cast<I*>(ix.data()),
                ix_size,
                static_cast<R*>(rel_types.data()),
                rel_types.size(),
                static_cast<size_t*>(output_memory.data()),
                static_cast<size_t*>(origin.data()),
                static_cast<I*>(adjacent.data()),
                static_cast<R*>(rel_adjacent.data()),
                static_cast<V*>(val_adjacent.data()),
                return_inner,
                return_relations,
                return_values
            );
            cudaDeviceSynchronize();
            maelstrom::cuda::cudaCheckErrors("k_query_adjacency");

            return std::make_tuple(
                std::move(origin),
                std::move(adjacent),
                std::move(val_adjacent),
                std::move(rel_adjacent)
            );
        }

        template <typename E, typename I, typename V>
        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency_device_dispatch_rel(E execution_policy,
                                                                                                                            maelstrom::vector& row,
                                                                                                                            maelstrom::vector& col,
                                                                                                                            maelstrom::vector& val,
                                                                                                                            maelstrom::vector& rel,
                                                                                                                            maelstrom::vector& ix,
                                                                                                                            maelstrom::vector& rel_types,
                                                                                                                            bool return_inner,
                                                                                                                            bool return_values,
                                                                                                                            bool return_relations) 
        {
            if(rel_types.empty()) {
                return exec_query_adjacency_device<E, I, V, uint8_t>(
                    execution_policy,
                    row,
                    col,
                    val,
                    rel,
                    ix,
                    rel_types,
                    return_inner,
                    return_values,
                    return_relations
                );
            }

            // only support uint8 for now
            switch(rel.get_dtype().prim_type) {
                case UINT8:
                    return exec_query_adjacency_device<E, I, V, uint8_t>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
            }

            throw std::runtime_error("unsupported relation type for query adjacency");
        }

        template <typename E, typename I>
        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency_device_dispatch_val(E execution_policy,
                                                                                                                            maelstrom::vector& row,
                                                                                                                            maelstrom::vector& col,
                                                                                                                            maelstrom::vector& val,
                                                                                                                            maelstrom::vector& rel,
                                                                                                                            maelstrom::vector& ix,
                                                                                                                            maelstrom::vector& rel_types,
                                                                                                                            bool return_inner,
                                                                                                                            bool return_values,
                                                                                                                            bool return_relations)
        {
            switch(val.get_dtype().prim_type) {
                case UINT64:
                    return query_adjacency_device_dispatch_rel<E, I, uint64_t>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
                case UINT32:
                    return query_adjacency_device_dispatch_rel<E, I, uint32_t>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
                case UINT8:
                    return query_adjacency_device_dispatch_rel<E, I, uint8_t>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
                case INT64:
                    return query_adjacency_device_dispatch_rel<E, I, int64_t>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
                case INT32:
                    return query_adjacency_device_dispatch_rel<E, I, int32_t>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
                case INT8:
                    return query_adjacency_device_dispatch_rel<E, I, int8_t>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
                case FLOAT64:
                    return query_adjacency_device_dispatch_rel<E, I, double>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
                case FLOAT32:
                    return query_adjacency_device_dispatch_rel<E, I, float>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
            }

            throw std::runtime_error("unsupported val type for query adjacency");
        }

        template <typename E>
        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency_device_dispatch_ix(E execution_policy,
                                                                                                                           maelstrom::vector& row,
                                                                                                                           maelstrom::vector& col,
                                                                                                                           maelstrom::vector& val,
                                                                                                                           maelstrom::vector& rel,
                                                                                                                           maelstrom::vector& ix,
                                                                                                                           maelstrom::vector& rel_types,
                                                                                                                           bool return_inner,
                                                                                                                           bool return_values,
                                                                                                                           bool return_relations)
        {
            switch(row.get_dtype().prim_type) {
                case UINT64:
                    return query_adjacency_device_dispatch_val<E, uint64_t>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
                case UINT32:
                    return query_adjacency_device_dispatch_val<E, uint32_t>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
                case UINT8:
                    return query_adjacency_device_dispatch_val<E, uint8_t>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
                case INT64:
                    return query_adjacency_device_dispatch_val<E, int64_t>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
                case INT32:
                    return query_adjacency_device_dispatch_val<E, int32_t>(
                        execution_policy,
                        row,
                        col,
                        val,
                        rel,
                        ix,
                        rel_types,
                        return_inner,
                        return_values,
                        return_relations
                    );
            }

            throw std::runtime_error("unsupported index type for query adjacency");
        }

        template
        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency_device_dispatch_ix(maelstrom::device_exec_t execution_policy,
                                                                                                                                maelstrom::vector& row,
                                                                                                                                maelstrom::vector& col,
                                                                                                                                maelstrom::vector& val,
                                                                                                                                maelstrom::vector& rel,
                                                                                                                                maelstrom::vector& ix,
                                                                                                                                maelstrom::vector& rel_types,
                                                                                                                                bool return_inner,
                                                                                                                                bool return_values,
                                                                                                                                bool return_relations);

    }
}