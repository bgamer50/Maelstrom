#include "maelstrom/storage/datatype.h"
#include <limits>
#include <cstdint>
#include <stdexcept>

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

    primitive_t prim_type_of(std::any a) {
        const std::type_info& t = a.type();
        if(t == typeid(uint64_t)) return primitive_t::UINT64;
        if(t == typeid(uint32_t)) return primitive_t::UINT32;
        if(t == typeid(uint8_t)) return primitive_t::UINT8;
        if(t == typeid(int64_t)) return primitive_t::INT64;
        if(t == typeid(int32_t)) return primitive_t::INT32;
        if(t == typeid(int8_t)) return primitive_t::INT8;
        if(t == typeid(double)) return primitive_t::FLOAT64;
        if(t == typeid(float)) return primitive_t::FLOAT32;

        throw std::runtime_error("Unknown type");
    }

    dtype_t dtype_from_prim_type(primitive_t prim_type) {
        switch(prim_type) {
            case UINT64:
               return uint64;
            case UINT32:
                return uint32;
            case UINT8:
                return uint8;
            case INT64:
                return int64;
            case INT32:
                return int32;
            case INT8:
                return int8;
            case FLOAT64:
                return float64;
            case FLOAT32:
                return float32;
        }

        throw std::runtime_error("Invalid primitive type");
    }

    std::pair<std::vector<unsigned char>, primitive_t> any_to_bytes(std::any& a) {
        primitive_t prim_type = prim_type_of(a);
        std::vector<unsigned char> bytes(size_of(prim_type));
        void* data = bytes.data();

        switch(prim_type) {
            case UINT64: {
                *static_cast<uint64_t*>(data) = std::any_cast<uint64_t>(a);
                break;
            }
            case UINT32: {
                *static_cast<uint32_t*>(data) = std::any_cast<uint32_t>(a);
                break;
            }
            case UINT8: {
                *static_cast<uint8_t*>(data) = std::any_cast<uint8_t>(a);
                break;
            }
            case INT64: {
                *static_cast<int64_t*>(data) = std::any_cast<int64_t>(a);
                break;
            }
            case INT32: {
                *static_cast<int32_t*>(data) = std::any_cast<int32_t>(a);
                break;
            }
            case INT8: {
                *static_cast<int8_t*>(data) = std::any_cast<uint8_t>(a);
                break;
            }
            case FLOAT64: {
                *static_cast<double*>(data) = std::any_cast<double>(a);
                break;
            }
            case FLOAT32: {
                *static_cast<float*>(data) = std::any_cast<float>(a);
                break;
            }
            default:
                throw std::runtime_error("invalid primitive type provided to any_to_bytes");
        }

        return std::make_pair(bytes, prim_type);
    }

    std::any max_value(dtype_t& dtype) {
        switch(dtype.prim_type) {
            case UINT64:
                return std::numeric_limits<uint64_t>::max();
            case UINT32:
                return std::numeric_limits<uint32_t>::max();
            case UINT8:
                return std::numeric_limits<uint8_t>::max();
            case INT64:
                return std::numeric_limits<int64_t>::max();
            case INT32:
                return std::numeric_limits<int32_t>::max();
            case INT8:
                return std::numeric_limits<int8_t>::max();
            case FLOAT64:
                return std::numeric_limits<double>::max();
            case FLOAT32:
                return std::numeric_limits<float>::max();
        }

        throw std::runtime_error("invalid dtype provided to max_value");
    }

    bool dtype_t::operator==(const dtype_t& other) {
        return (other.name == this->name) && (other.prim_type == this->prim_type);
    }

    bool dtype_t::operator!=(const dtype_t& other) {
        return (other.name != this->name) || (other.prim_type != this->prim_type);
    }

    // The serializers below may seem redundant, but are necessary because they
    // will throw the correct error when a type conversion needs to occur.
    // Otherwise, a bad_any_cast might get thrown.

    dtype_t uint64{
        "uint64",
        primitive_t::UINT64,
        [](void* v){ return std::any(*static_cast<uint64_t*>(v)); },
        [](std::any a){ return std::any(std::any_cast<uint64_t>(a)); }
    };

    dtype_t uint32{
        "uint32",
        primitive_t::UINT32,
        [](void* v){ return std::any(*static_cast<uint32_t*>(v)); },
        [](std::any a){ return std::any(std::any_cast<uint32_t>(a)); }
    };

    dtype_t uint8{
        "uint8",
        primitive_t::UINT8,
        [](void* v){ return std::any(*static_cast<uint8_t*>(v)); },
        [](std::any a){ return std::any(std::any_cast<uint8_t>(a)); }
    };

    dtype_t int64{
        "int64",
        primitive_t::INT64,
        [](void* v){ return std::any(*static_cast<int64_t*>(v)); },
        [](std::any a){ return std::any(std::any_cast<int64_t>(a)); }
    };

    dtype_t int32{
        "int32",
        primitive_t::INT32,
        [](void* v){ return std::any(*static_cast<int32_t*>(v)); },
        [](std::any a){ return std::any(std::any_cast<int32_t>(a)); }
    };

    dtype_t int8{
        "int8",
        primitive_t::INT8,
        [](void* v){ return std::any(*static_cast<int8_t*>(v)); },
        [](std::any a){ return std::any(std::any_cast<int8_t>(a)); }
    };

    dtype_t float64{
        "float64",
        primitive_t::FLOAT64,
        [](void* v){ return std::any(*static_cast<double*>(v)); },
        [](std::any a){ return std::any(std::any_cast<double*>(a)); }
    };

    dtype_t float32{
        "float32",
        primitive_t::FLOAT32,
        [](void* v){ return std::any(*static_cast<float*>(v)); },
        [](std::any a){ return std::any(std::any_cast<float*>(a)); }
    };

    dtype_t default_dtype = float64;
}