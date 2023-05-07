#include "storage/datatype.h"

namespace maelstrom {

    size_t size_of(primitive_t& prim_type) {
        switch(prim_type) {
            case UINT64:
                return sizeof(uint64_t);
            case UINT32:
                return sizeof(uint32_t);
            case UINT8:
                return sizeof(uint8_t);
            case INT64:
                return sizeof(int64_t);
            case INT32:
                return sizeof(int32_t);
            case INT8:
                return sizeof(int8_t);
            case FLOAT64:
                return sizeof(double);
            case FLOAT32:
                return sizeof(float);
        }

        throw std::runtime_error("Invalid type");
    }

    dtype_t uint64{
        "uint64",
        primitive_t::UINT64,
        [](void* v){ return boost::any(*static_cast<uint64_t*>(v)); }
    };

    dtype_t uint32{
        "uint32",
        primitive_t::UINT32,
        [](void* v){ return boost::any(*static_cast<uint32_t*>(v)); }
    };

    dtype_t uint8{
        "uint8",
        primitive_t::UINT8,
        [](void* v){ return boost::any(*static_cast<uint8_t*>(v)); }
    };

    dtype_t int64{
        "int64",
        primitive_t::INT64,
        [](void* v){ return boost::any(*static_cast<int64_t*>(v)); }
    };

    dtype_t int32{
        "int32",
        primitive_t::INT32,
        [](void* v){ return boost::any(*static_cast<int32_t*>(v)); }
    };

    dtype_t int8{
        "int8",
        primitive_t::INT8,
        [](void* v){ return boost::any(*static_cast<int8_t*>(v)); }
    };

    dtype_t float64{
        "float64",
        primitive_t::FLOAT64,
        [](void* v){ return boost::any(*static_cast<double*>(v)); }
    };

    dtype_t float32{
        "float32",
        primitive_t::FLOAT32,
        [](void* v){ return boost::any(*static_cast<float*>(v)); }
    };
}