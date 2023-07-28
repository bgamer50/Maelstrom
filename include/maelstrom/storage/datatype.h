#pragma once

#include <any>
#include <functional>
#include <string>

namespace maelstrom {
    enum primitive_t {UINT64=0, UINT32=1, INT64=2, INT32=3, FLOAT64=4, FLOAT32=5, UINT8=6, INT8=7};
    
    struct dtype_t {
        std::string name;
        primitive_t prim_type;
        std::function<std::any(void* data)> deserialize;
        std::function<std::any(std::any)> serialize = [](std::any a){return a;};

        bool operator==(const dtype_t& other);
        bool operator!=(const dtype_t& other);
    };

    size_t size_of(primitive_t& prim_type);

    inline size_t size_of(dtype_t& dtype) { return size_of(dtype.prim_type); }

    primitive_t prim_type_of(std::any a);

    dtype_t dtype_from_prim_type(primitive_t prim_type);

    std::pair<std::vector<unsigned char>, primitive_t> any_to_bytes(std::any& a);

    std::any max_value(maelstrom::dtype_t& dtype);

    extern dtype_t uint64;
    extern dtype_t uint32;
    extern dtype_t uint8;
    extern dtype_t int64;
    extern dtype_t int32;
    extern dtype_t int8;
    extern dtype_t float64;
    extern dtype_t float32;

    extern dtype_t default_dtype;
}