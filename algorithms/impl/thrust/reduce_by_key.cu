#include "algorithms/reduce_by_key.h"
#include "algorithms/sort.h"
#include "algorithms/select.h"

template <typename E, typename K, typename V>
std::pair<maelstrom::vector, maelstrom::vector> t_reduce_by_key_from_sorted_vec(E exec_policy, maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, size_t max_unique_keys) {
    
}

template <typename E, typename K>
std::pair<maelstrom::vector, maelstrom::vector> reduce_by_key_from_sorted_vec_dispatch_val(E exec_policy, maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, size_t max_unique_keys) {
    
}

template <typename E>
std::pair<maelstrom::vector, maelstrom::vector> reduce_by_key_from_sorted_vec_dispatch_key(E exec_policy, maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, size_t max_unique_keys) {
    
}

std::pair<maelstrom::vector, maelstrom::vector> reduce_by_key_from_sorted_vec_dispatch_exec_policy(maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, size_t max_unique_keys) {

}

std::pair<maelstrom::vector, maelstrom::vector> reduce_by_key(maelstrom::vector& input_keys, maelstrom::vector& input_values, maelstrom::reductor red, size_t max_unique_keys, bool sorted=false) {
    if(sorted) {
        return reduce_by_key_from_sorted_vec_dispatch_exec_policy(input_keys, input_values, red, max_unique_keys);
    }

    maelstrom::vector input_keys_copy(input_keys);

    auto ix = maelstrom::sort(input_keys_copy);
    maelstrom::vector input_values_copy = maelstrom::select(input_values, ix);
    
    maelstrom::vector result_keys;
    maelstrom::vector result_vals;

    std::tie(result_keys, result_vals) = reduce_by_key_from_sorted_vec_dispatch_exec_policy(
        input_keys_copy,
        input_values_copy,
        red,
        max_unique_keys
    );

    result_keys = maelstrom::select(ix, result_keys);

    return std::make_pair(
        std::move(result_keys),
        std::move(result_vals)
    );
}