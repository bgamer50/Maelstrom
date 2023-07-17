#pragma once

#include "cuco/utility/traits.hpp"
#include "cuco/utility/allocator.hpp"
#include "maelstrom/storage/allocators.cuh"

CUCO_DECLARE_BITWISE_COMPARABLE(double)
CUCO_DECLARE_BITWISE_COMPARABLE(float)

namespace maelstrom {

    typedef cuco::cuda_allocator<char> device_allocator_t;
    typedef maelstrom::maelstrom_managed_allocator<char> managed_allocator_t;

}