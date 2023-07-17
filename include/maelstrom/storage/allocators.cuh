#pragma once
#include "maelstrom/util/cuda_utils.cuh"

namespace maelstrom {

    template <typename T>
    class maelstrom_managed_allocator {
        public:
            using value_type = T;

            maelstrom_managed_allocator() = default;

            // copy constructor
            template <class C>
            maelstrom_managed_allocator(maelstrom_managed_allocator<C> const&) noexcept {}

            // allocates storage for n objects of type T in managed memory
            value_type* allocate(std::size_t n) {
                value_type* p;
                cudaMallocManaged(&p, sizeof(value_type) * n);
                cudaDeviceSynchronize();
                maelstrom::cuda::cudaCheckErrors("allocate managed memory");

                return p;
            }

            // deallocates using cudaFree
            void deallocate(value_type* p, size_t) {
                cudaFree(p);
                cudaDeviceSynchronize();
                maelstrom::cuda::cudaCheckErrors("deallocate managed memory");
            }
    };

    template <typename T, typename U>
    bool operator==(maelstrom_managed_allocator<T> const&, maelstrom_managed_allocator<U> const&) noexcept {
        return true;
    }

    template <typename T, typename U>
    bool operator!=(maelstrom_managed_allocator<T> const& lhs, maelstrom_managed_allocator<U> const& rhs) noexcept {
        return !(lhs == rhs);
    }

    template <typename T>
    class maelstrom_host_allocator {
        public:
            using value_type = T;

            maelstrom_host_allocator() = default;

            // copy constructor
            template <class C>
            maelstrom_host_allocator(maelstrom_host_allocator<C> const&) noexcept {}

            // allocates storage for n objects of type T in managed memory
            value_type* allocate(std::size_t n) {
                value_type* p;
                cudaMallocManaged(&p, sizeof(value_type) * n);
                cudaMemAdvise(p, sizeof(value_type) * n, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
                cudaDeviceSynchronize();
                maelstrom::cuda::cudaCheckErrors("allocate host memory");

                return p;
            }

            // deallocates using cudaFree
            void deallocate(value_type* p, size_t) {
                cudaFree(p);
                cudaDeviceSynchronize();
                maelstrom::cuda::cudaCheckErrors("deallocate host memory");
            }
    };

    template <typename T, typename U>
    bool operator==(maelstrom_host_allocator<T> const&, maelstrom_host_allocator<U> const&) noexcept {
        return true;
    }

    template <typename T, typename U>
    bool operator!=(maelstrom_host_allocator<T> const& lhs, maelstrom_host_allocator<U> const& rhs) noexcept {
        return !(lhs == rhs);
    }

}