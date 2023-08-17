#include "maelstrom/containers/hash_table.h"

#include <sstream>

namespace maelstrom {

    hash_table::hash_table(maelstrom::storage mem_type, maelstrom::dtype_t key_dtype, maelstrom::dtype_t val_dtype, size_t initial_size) {
        this->mem_type = mem_type;
        this->key_dtype = key_dtype;
        this->val_dtype = val_dtype;

        switch(this->mem_type) {
            case HOST:
                this->data_ptr = maelstrom::instantiate_hash_table<HOST>(key_dtype, val_dtype, initial_size);
                break;
            case DEVICE:
                this->data_ptr = maelstrom::instantiate_hash_table<DEVICE>(key_dtype, val_dtype, initial_size);
                break;
            case MANAGED:
                this->data_ptr = maelstrom::instantiate_hash_table<MANAGED>(key_dtype, val_dtype, initial_size);
                break;
            default:
                throw std::runtime_error("Unsupported memory type for hash table");
        }
    }

    hash_table::~hash_table() noexcept(false) {
        switch(this->mem_type) {
            case HOST:
                maelstrom::destroy_hash_table<HOST>(this->data_ptr, this->key_dtype, this->val_dtype);
                break;
            case DEVICE:
                maelstrom::destroy_hash_table<DEVICE>(this->data_ptr, this->key_dtype, this->val_dtype);
                break;
            case MANAGED:
                maelstrom::destroy_hash_table<MANAGED>(this->data_ptr, this->key_dtype, this->val_dtype);
                break;
            default:
                throw std::runtime_error("Unsupported memory type for hash table deletion");
        }
    }

    void hash_table::set(maelstrom::vector& keys, maelstrom::vector& vals) {
        if(this->mem_type == HOST && (keys.get_mem_type() == DEVICE || vals.get_mem_type() == DEVICE)) {
            throw std::invalid_argument("can't use device array to set host hash table");
        }

        if(keys.get_dtype() != this->key_dtype) {
            std::stringstream sx;
            sx << "key dtype must match! (got " << keys.get_dtype().name << " but expected " << this->key_dtype.name << ")";
            throw std::invalid_argument(sx.str());
        } 
        
        if(vals.get_dtype() != this->val_dtype) {
            std::stringstream sx;
            sx << "val dtype must match! (got " << vals.get_dtype().name << " but expected " << this->val_dtype.name << ")";
            throw std::invalid_argument(sx.str());
        }

        if(keys.size() != vals.size()) throw std::invalid_argument("number of keys must match number of values for insert!");

        switch(this->mem_type) {
            case HOST:
                maelstrom::insert_hash_table<HOST>(this->data_ptr, keys, vals);
                break;
            case DEVICE:
                maelstrom::insert_hash_table<DEVICE>(this->data_ptr, keys, vals);
                break;
            case MANAGED:
                maelstrom::insert_hash_table<MANAGED>(this->data_ptr, keys, vals);
                break;
            default:
                throw std::runtime_error("unsupported memory type for hash table insertion");
        }
    }

    maelstrom::vector hash_table::get(maelstrom::vector& keys) {
        if(this->mem_type == HOST && (keys.get_mem_type() == DEVICE)) {
            throw std::runtime_error("can't use device array to access host hash table");
        }
        if(keys.get_dtype() != this->key_dtype) throw std::runtime_error("key dtype must match!");

        switch(this->mem_type) {
            case HOST:
                return maelstrom::get_hash_table<HOST>(this->data_ptr, keys, this->val_dtype, true);
            case DEVICE:
                return maelstrom::get_hash_table<DEVICE>(this->data_ptr, keys, this->val_dtype, true);
            case MANAGED:
                return maelstrom::get_hash_table<MANAGED>(this->data_ptr, keys, this->val_dtype, true);
            default:
                throw std::runtime_error("unsupported memory type for hash table access");
        }
    }

    void hash_table::remove(maelstrom::vector& keys) {
        if(this->mem_type == HOST && (keys.get_mem_type() == DEVICE)) {
            throw std::runtime_error("can't use device array to access host hash table");
        }
        if(keys.get_dtype() != this->key_dtype) throw std::runtime_error("key dtype must match!");

        switch(this->mem_type) {
            case HOST:
                return maelstrom::remove_hash_table<HOST>(this->data_ptr, keys, this->val_dtype);
            case DEVICE:
                return maelstrom::remove_hash_table<DEVICE>(this->data_ptr, keys, this->val_dtype);
            case MANAGED:
                return maelstrom::remove_hash_table<MANAGED>(this->data_ptr, keys, this->val_dtype);
            default:
                throw std::runtime_error("unsupported memory type for hash table access");
        }
    }

    maelstrom::vector hash_table::contains(maelstrom::vector& keys) {
        if(this->mem_type == HOST && (keys.get_mem_type() == DEVICE)) {
            throw std::runtime_error("can't use device array to access host hash table");
        }
        if(keys.get_dtype() != this->key_dtype) throw std::runtime_error("key dtype must match!");

        switch(this->mem_type) {
            case HOST:
                return maelstrom::get_hash_table<HOST>(this->data_ptr, keys, this->val_dtype, false);
            case DEVICE:
                return maelstrom::get_hash_table<DEVICE>(this->data_ptr, keys, this->val_dtype, false);
            case MANAGED:
                return maelstrom::get_hash_table<MANAGED>(this->data_ptr, keys, this->val_dtype, false);
            default:
                throw std::runtime_error("unsupported memory type for hash table access");
        }
    }


}