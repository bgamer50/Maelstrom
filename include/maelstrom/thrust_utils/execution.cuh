#pragma once

#include "containers/vector.h"
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

#include <optional>
#include <boost/any.hpp>

namespace maelstrom {

    typedef thrust::system::cuda::detail::par_t device_exec_t;
    typedef thrust::system::cpp::detail::par_t host_exec_t;

    class exec_policy {
        private:
            std::optional<device_exec_t> device_exec_policy = std::nullopt;
            std::optional<host_exec_t> host_exec_policy = std::nullopt;
        public:
            exec_policy(bool device=true) {
                if(device) this->device_exec_policy.emplace(thrust::device);
                else this->host_exec_policy.emplace(thrust::host);
            }

            exec_policy(device_exec_t device_policy) {
                this->device_exec_policy.emplace(device_policy);
            }

            exec_policy(host_exec_t host_policy) {
                this->host_exec_policy.emplace(host_policy);
            }

            boost::any get() {
                if(device_exec_policy) return device_exec_policy.value();
                if(host_exec_policy) return host_exec_policy.value();
                throw std::runtime_error("no valid exec policy!");
            }
    };

    inline exec_policy get_execution_policy(maelstrom::vector& vec) {
        switch(vec.get_mem_type()) {
            case HOST:
            case PINNED:
                return exec_policy(false);
            case DEVICE:
            case MANAGED:
                return exec_policy(true);
        }

        throw std::runtime_error("Illegal memory type!");
    }

}