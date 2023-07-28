#pragma once

#include "maelstrom/storage/datatype.h"

#include <string>
#include <sstream>
#include <unordered_set>

namespace maelstrom {

    template<typename T>
    struct string_index_hash{
        size_t operator()(const std::pair<std::string, T>& key) const {
            return std::hash<size_t>()(
                std::hash<std::string>()(key.first) * std::hash<T>()(key.second)
            );
        }
    };

    template<typename T>
    struct string_index_equals{
        bool operator()(const std::pair<std::string, T>& left, const std::pair<std::string, T>& right) const {
            return (left.first == right.first) || (left.second == right.second);
        }
    };

    template<typename T>
    class string_index {
        private:
            std::unordered_set<std::pair<std::string, T>, string_index_hash<T>, string_index_equals<T>> data;
            
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

            inline size_t encode(std::string str) { 
                auto it = this->data.find(std::make_pair(str, invalid_repr));
                if(it == data.end()) {
                    if(next_repr == invalid_repr) throw std::runtime_error("reached size limit");
                    auto repr = next_repr;
                    this->data.insert(std::make_pair(str, repr));
                    ++next_repr;
                    return repr;
                }
                
                return it->second;
            }

            inline std::string decode(T repr) {
                auto it = this->data.find(std::make_pair(invalid_str, repr));
                if(it == data.end()) {
                    std::stringstream sx;
                    sx << "No string exists with given representation: " << repr;
                    throw std::runtime_error(sx.str());
                }

                return it->first;
            }

            inline dtype_t get_dtype() {
                return this->dtype;
            }
    };

}