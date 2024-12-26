#pragma once

#include <unordered_map>
#include <string>
#include <any>
#include <sstream>

namespace maelstrom
{
    enum storage {
        HOST=0,
        DEVICE=1,
        MANAGED=2,
        PINNED=3,
        DIST_DEVICE=4,
        DIST_HOST=5,
        DIST_MANAGED=6
    };

    extern std::unordered_map<std::string, maelstrom::storage> storage_string_mapping;

    inline bool is_dist(storage s) {
        return (
            (s == storage::DIST_DEVICE) ||
            (s == storage::DIST_HOST) ||
            (s == storage::DIST_MANAGED)
        );
    }

    inline storage dist_storage_of(storage s) {
        switch(s) {
            case HOST:
            case DIST_HOST:
                return DIST_HOST;
            case DEVICE:
            case DIST_DEVICE:
                return DIST_DEVICE;
            case MANAGED:
            case DIST_MANAGED:
                return DIST_MANAGED;
        }

        std::stringstream msg;
        msg << "Storage type " << s << " is not supported for distributed processing.";
        throw std::runtime_error(msg.str());
    }

    inline storage single_storage_of(storage s) {
        switch(s) {
            case HOST:
            case DIST_HOST:
                return HOST;
            case DEVICE:
            case DIST_DEVICE:
                return DEVICE;
            case MANAGED:
            case DIST_MANAGED:
                return MANAGED;
            case PINNED:
                return PINNED;
        }

        std::stringstream msg;
        msg << "Invalid storage type " << s;
        throw std::runtime_error(msg.str());
    }

    /*
        Creates a new stream associated with the given storage.
        i.e. calls cudaStreamCreate for DEVICE storage.
        If the storage doesn't support streams, an empty
        any is returned instead.
    */
    std::any create_stream(storage s);

    /*
        Destroys the stream associated with the given storage.
    */
    void destroy_stream(storage s, std::any stream);

    std::any get_default_stream(storage s);
}
