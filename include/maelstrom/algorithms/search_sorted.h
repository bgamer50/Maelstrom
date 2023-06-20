#pragma once

#include "maelstrom/containers/vector.h"

namespace maelstrom {

    /*
        Returns the position such that if the value were inserted there,
        the array would remain sorted.  Returns the last such index.
        If the value is smaller than the first element, 0 is returned.
        If the value is larger than that last element, the size of the sorted
        array is returned.
    */
    maelstrom::vector search_sorted(maelstrom::vector& sorted_array, maelstrom::vector& values_to_find);

}