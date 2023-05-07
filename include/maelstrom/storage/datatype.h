#pragma once

#include <boost/any.hpp>
#include <functional>
#include <string>

namespace maelstrom {
    enum primitive_t {UINT64=0, UINT32=1, INT64=2, INT32=3, FLOAT64=4, FLOAT32=5, UINT8=6, INT8=7};
    
    struct dtype_t {
        std::string name;
        primitive_t prim_type;
        std::function<boost::any(void* data)> deserialize;
        std::function<boost::any(boost::any)> serialize = [](boost::any a){return a;};
    };

    size_t size_of(primitive_t& prim_type);

    inline size_t size_of(dtype_t& dtype) { return size_of(dtype.prim_type); }

    extern dtype_t uint64;
    extern dtype_t uint32;
    extern dtype_t uint8;
    extern dtype_t int64;
    extern dtype_t int32;
    extern dtype_t int8;
    extern dtype_t float64;
    extern dtype_t float32;
}