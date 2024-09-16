#pragma once

#include "maelstrom/containers/vector.h"
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

#include <optional>
#include <any>

namespace maelstrom {

    typedef thrust::cuda_cub::par_t::stream_attachment_type device_exec_t;
    typedef thrust::system::cpp::detail::par_t host_exec_t;

    class exec_policy {
        private:
            std::optional<device_exec_t> device_exec_policy = std::nullopt;
            std::optional<host_exec_t> host_exec_policy = std::nullopt;
        public:
            exec_policy(bool device=true, std::optional<cudaStream_t> stream=std::nullopt) {
                if(device) {
                    this->device_exec_policy.emplace(
                        thrust::cuda::par.on(
                            stream.value_or((cudaStream_t)cudaStreamDefault)
                        )
                    );
                }
                else this->host_exec_policy.emplace(thrust::host);
            }

            exec_policy(device_exec_t device_policy) {
                this->device_exec_policy.emplace(device_policy);
            }

            exec_policy(host_exec_t host_policy) {
                this->host_exec_policy.emplace(host_policy);
            }

            std::any get() {
                if(device_exec_policy) return device_exec_policy.value();
                if(host_exec_policy) return host_exec_policy.value();
                throw std::runtime_error("no valid exec policy!");
            }
    };

    inline exec_policy get_execution_policy(maelstrom::vector& vec) {
        auto mem_type = vec.get_mem_type();
        switch(maelstrom::single_storage_of(mem_type)) {
            case HOST:
                return exec_policy(false);
            case DEVICE:
            case PINNED:
            case MANAGED: {
                std::optional<cudaStream_t> st = std::nullopt;
                auto stream = vec.get_stream();
                if(stream.has_value()) st.emplace(std::any_cast<cudaStream_t>(stream));
                return exec_policy(true, st);
            }
        }

        throw std::runtime_error("Illegal memory type!");
    }

}