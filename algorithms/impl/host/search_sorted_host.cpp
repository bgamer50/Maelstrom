#include "maelstrom/algorithms/search_sorted.h"
#include <iostream>

namespace maelstrom {

    template <typename T>
    maelstrom::vector h_search_sorted(maelstrom::vector& sorted_array_vector, maelstrom::vector& values_to_find_vector) {
        T* sorted_array = static_cast<T*>(sorted_array_vector.data());
        T* values_to_find = static_cast<T*>(values_to_find_vector.data());

        maelstrom::vector output_vector(values_to_find_vector.get_mem_type(), uint64, values_to_find_vector.size(), values_to_find_vector.local_size());
        size_t* output = static_cast<size_t*>(output_vector.data());

        // TODO parallelize
        for(size_t k = 0; k < values_to_find_vector.local_size(); ++k) {
            const T value = values_to_find[k];

            size_t left = 0;
            size_t right = sorted_array_vector.size();
            
            while(left < right) {
                const size_t i = (left + right) / 2;
                const T lower_value = sorted_array[i];

                if(lower_value <= value) left = i + 1;
                else right = i;

            }

            output[k] = left;
        }

        return output_vector;
    }

    maelstrom::vector search_sorted_host_dispatch_val(maelstrom::vector& sorted_array, maelstrom::vector& values_to_find) {
        switch(sorted_array.get_dtype().prim_type) {
            case UINT64:
                return h_search_sorted<uint64_t>(sorted_array, values_to_find);
            case UINT32:
                return h_search_sorted<uint32_t>(sorted_array, values_to_find);
            case UINT8:
                return h_search_sorted<uint8_t>(sorted_array, values_to_find);
            case INT64:
                return h_search_sorted<int64_t>(sorted_array, values_to_find);
            case INT32:
                return h_search_sorted<int32_t>(sorted_array, values_to_find);
            case INT8:
                return h_search_sorted<int8_t>(sorted_array, values_to_find);
            case FLOAT64:
                return h_search_sorted<double>(sorted_array, values_to_find);
            case FLOAT32:
                return h_search_sorted<float>(sorted_array, values_to_find);
        }

        throw std::runtime_error("Invalid data type passed to search_sorted");
    }

}