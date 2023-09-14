#pragma once

#include "maelstrom/storage/datatype.h"

#include <string>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include <type_traits>

namespace maelstrom {

    template<typename T>
    class string_index {
        private:
            std::vector<std::string> repr_to_str;
            std::unordered_map<std::string, T> str_to_repr;
            
            std::string invalid_str;
            T invalid_repr;
            
            T next_repr;

            maelstrom::dtype_t dtype;

        public:
            string_index(std::string invalid_str, T invalid_repr) {
                this->invalid_str = invalid_str;
                this->invalid_repr = invalid_repr;
                this->next_repr = static_cast<T>(0);

                auto prim_type = maelstrom::prim_type_of(next_repr);
                this->dtype = maelstrom::dtype_t{
                    "string",
                    prim_type,
                    [this](void* v){ return this->decode(*static_cast<T*>(v)); },
                    [this](std::any a) { return this->encode(std::any_cast<std::string>(a)); }
                };
            }

            inline T encode(std::string str) { 
                auto it = this->str_to_repr.find(str);
                if(it == str_to_repr.end()) {
                    auto repr = next_repr;
                    ++next_repr;

                    str_to_repr[str] = repr;
                    repr_to_str.push_back(str);
                    return repr;
                }

                return it->second;
            }

            inline std::string decode(T repr) {
                if((std::is_signed_v<T> && repr < 0) || repr > this->repr_to_str.size()) {
                    std::stringstream sx;
                    sx << "No string exists with given representation: " << repr;
                    throw std::runtime_error(sx.str());
                }
            
                return this->repr_to_str[repr];
            }

            inline dtype_t get_dtype() {
                return this->dtype;
            }
    };

}