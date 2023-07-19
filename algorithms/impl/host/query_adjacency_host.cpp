#include "maelstrom/containers/vector.h"

namespace maelstrom {
    namespace sparse {

        template <typename I, typename R>
        void h_query_adjacency_get_mem(I* row, I* col, R* rel, I* query, size_t query_size, R* query_rel, size_t query_rel_size, size_t* output_memory) {            
            const bool filter_rel = query_rel_size > 0;

            // TODO parallelize
            for(size_t i = 0; i < query_size; ++i) {
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
        void h_query_adjacency(I* row, I* col, R* rel, V* val, I* query, size_t query_size, R* query_rel, size_t query_rel_size, size_t* ps, size_t* origin, I* adjacent, R* rel_adjacent, V* val_adjacent, bool return_adj, bool return_rel, bool return_val) {            
            const bool filter_rel = query_rel_size > 0;

            for(size_t i = 0; i < query_size; ++i) {
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

        template <typename I, typename V, typename R>
        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> exec_query_adjacency_host(maelstrom::vector& row,
                                                                                                                         maelstrom::vector& col,
                                                                                                                         maelstrom::vector& val,
                                                                                                                         maelstrom::vector& rel,
                                                                                                                         maelstrom::vector& ix,
                                                                                                                         maelstrom::vector& rel_types,
                                                                                                                         bool return_inner,
                                                                                                                         bool return_values,
                                                                                                                         bool return_relations)
        {
            maelstrom::vector output_memory(
                row.get_mem_type(),
                maelstrom::uint64,
                ix.size()
            );

            h_query_adjacency_get_mem(
                static_cast<I*>(row.data()),
                static_cast<I*>(col.data()),
                static_cast<R*>(rel.data()),
                static_cast<I*>(ix.data()),
                ix.size(),
                static_cast<R*>(rel_types.data()),
                rel_types.size(),
                static_cast<size_t*>(output_memory.data())
            );

            // TODO parallelize
            for(size_t k = 1; k < output_memory.size(); ++k) static_cast<size_t*>(output_memory.data())[k] += static_cast<size_t*>(output_memory.data())[k-1];

            size_t output_size = std::any_cast<size_t>(output_memory.get(output_memory.size() - 1));

            maelstrom::vector origin(row.get_mem_type(), uint64, output_size);
            maelstrom::vector adjacent(row.get_mem_type(), row.get_dtype(), return_inner ? output_size : 0);
            maelstrom::vector rel_adjacent(row.get_mem_type(), rel.get_dtype(), return_relations ? output_size : 0);
            maelstrom::vector val_adjacent(row.get_mem_type(), val.get_dtype(), return_values ? output_size : 0);

            h_query_adjacency(
                static_cast<I*>(row.data()),
                static_cast<I*>(col.data()),
                static_cast<R*>(rel.data()),
                static_cast<V*>(val.data()),
                static_cast<I*>(ix.data()),
                ix.size(),
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

            return std::make_tuple(
                std::move(origin),
                std::move(adjacent),
                std::move(val_adjacent),
                std::move(rel_adjacent)
            );
        }

        template <typename I, typename V>
        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency_host_dispatch_rel(maelstrom::vector& row,
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
                return exec_query_adjacency_host<I, V, uint8_t>(
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
                    return exec_query_adjacency_host<I, V, uint8_t>(
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

        template <typename I>
        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency_host_dispatch_val(maelstrom::vector& row,
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
                    return query_adjacency_host_dispatch_rel<I, uint64_t>(
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
                    return query_adjacency_host_dispatch_rel<I, uint32_t>(
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
                    return query_adjacency_host_dispatch_rel<I, uint8_t>(
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
                    return query_adjacency_host_dispatch_rel<I, int64_t>(
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
                    return query_adjacency_host_dispatch_rel<I, int32_t>(
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
                    return query_adjacency_host_dispatch_rel<I, int8_t>(
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
                    return query_adjacency_host_dispatch_rel<I, double>(
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
                    return query_adjacency_host_dispatch_rel<I, float>(
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

        std::tuple<maelstrom::vector, maelstrom::vector, maelstrom::vector, maelstrom::vector> query_adjacency_host_dispatch_ix(maelstrom::vector& row,
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
                    return query_adjacency_host_dispatch_val<uint64_t>(
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
                    return query_adjacency_host_dispatch_val<uint32_t>(
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
                    return query_adjacency_host_dispatch_val<uint8_t>(
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
                    return query_adjacency_host_dispatch_val<int64_t>(
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
                    return query_adjacency_host_dispatch_val<int32_t>(
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

    }
}