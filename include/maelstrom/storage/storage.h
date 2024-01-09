#pragma once

#include <unordered_map>
#include <string>

namespace maelstrom
{
    enum storage {HOST=0, DEVICE=1, MANAGED=2, PINNED=3};

    extern std::unordered_map<std::string, maelstrom::storage> storage_string_mapping;
}
