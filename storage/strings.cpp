#include "maelstrom/storage/strings.h"

namespace maelstrom {
    template class string_index<uint64_t>;
    template class string_index<uint32_t>;
    template class string_index<uint8_t>;
    template class string_index<int64_t>;
    template class string_index<int32_t>;
    template class string_index<int8_t>;
}