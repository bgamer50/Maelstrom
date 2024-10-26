#include "maelstrom/algorithms/similarity.h"
#include "maelstrom/thrust_utils/execution.cuh"
#include "maelstrom/util/cuda_utils.cuh"
#include "maelstrom/thrust_utils/thrust_utils.cuh"

namespace maelstrom {

    /*
        Each work unit is applied to a vector in the list of source embeddings.
        Each source embedding vector is compared to each of the target vectors,
        and the highest similarity score is returned.

        Parameters:
        -----------
        src_embeddings
            Contiguous 2D embedding tensor.
        src_offsets
            Offsets into the 2D embedding tensor; or, if nullptr, then
            assumed to be a range from 0 to (n_src - 1) which means the
            first n_src embeddings in src_embeddings will be searched.
        n_src
            The number of embeddings to search.  Should correspond
            to the size or implied size of src_offsets.
        tar_embeddings
            Contiguous 2D embedding tensor of embeddings to match.
        tar_magnitudes
            Square of the magnitude of each target embedding.
        n_tar
            Number of embeddings in tar_embeddings.
        buffer
            Temporary buffer of size n_src * n_tar
        dest
            Output buffer of size n_src that will receive the output
            cosine similarities.
        emb_stride
            Trailing dimension of src_embeddings and tar_embeddings,
            or in other words, the length of each embedding.
        
    */
    template<typename T>
    __global__ void k_cosine_similarity(T* src_embeddings, size_t* src_offsets, size_t n_src, T* tar_embeddings, T* tar_magnitudes, size_t n_tar, T* buffer, double* dest, size_t emb_stride) {
        const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = blockDim.x * gridDim.x;

        for(size_t k = index; k < n_src; k += stride) {
            const size_t offset = (src_offsets == nullptr) ? k : src_offsets[k];
            T* src_k = src_embeddings + (emb_stride * offset);
            T* sum = buffer + (k * n_tar);
            
            T mag_s = src_k[0] * src_k[0];
            
            for(size_t i = 0; i < n_tar; ++i) {
                T* tar_i = tar_embeddings + (emb_stride * i);
                sum[i] = tar_i[0] * src_k[0];
            }
            for(size_t n = 1; n < emb_stride; ++n) {
                const T c = src_k[n];
                mag_s += c * c;
                for(size_t i = 0; i < n_tar; ++i) {
                    T* tar_i = tar_embeddings + (emb_stride * i);
                    sum[i] += c * tar_i[n];
                }
            }

            dest[k] = sum[0] / sqrt(mag_s * tar_magnitudes[0]);
            for(size_t i = 1; i < n_tar; ++i) {
                double sim = sum[i] / sqrt(mag_s * tar_magnitudes[i]);
                dest[k] = (sim > dest[k]) ? sim : dest[k];
            }
        }
    }

    template <typename E, typename T>
    maelstrom::vector launch_similarity(E exec_policy, similarity_t metric, maelstrom::vector& src_embeddings, maelstrom::vector& src_offsets, maelstrom::vector& target_embeddings, size_t emb_stride) {
        if(metric == maelstrom::COSINE) {
            const size_t num_tar = target_embeddings.size() / emb_stride;
            
            // calculate the square of the magnitudes for each target
            maelstrom::vector magnitudes(maelstrom::PINNED, target_embeddings.get_dtype(), num_tar);
            for(size_t i = 0; i < num_tar; ++i) {
                static_cast<T*>(magnitudes.data())[i] = thrust::transform_reduce(
                    maelstrom::device_tptr_cast<T>(target_embeddings.data()) + (emb_stride * i),
                    maelstrom::device_tptr_cast<T>(target_embeddings.data()) + (emb_stride * (i+1)),
                    thrust::square<T>(),
                    (T)0,
                    thrust::plus<T>()
                );
            }
            magnitudes = magnitudes.to(maelstrom::DEVICE);            

            
            const size_t sz = src_offsets.empty() ? (src_embeddings.size() / emb_stride) : src_offsets.size();
            maelstrom::vector dest(src_embeddings.get_mem_type(), maelstrom::float64, sz);
            maelstrom::vector buffer(maelstrom::DEVICE, src_embeddings.get_dtype(), sz * num_tar);

            // launch kernel
            const size_t num_blocks = maelstrom::cuda::num_blocks(sz, MAELSTROM_DEFAULT_BLOCK_SIZE);
            k_cosine_similarity<<<num_blocks, MAELSTROM_DEFAULT_BLOCK_SIZE>>>(
                static_cast<T*>(src_embeddings.data()),
                src_offsets.empty() ? nullptr : static_cast<size_t*>(src_offsets.data()),
                sz,
                static_cast<T*>(target_embeddings.data()),
                static_cast<T*>(magnitudes.data()),
                num_tar,
                static_cast<T*>(buffer.data()),
                static_cast<double*>(dest.data()),
                emb_stride
            );
            cudaDeviceSynchronize();
            maelstrom::cuda::cudaCheckErrors("k_cosine_similarity");
            
            return dest;
        } else {
            throw std::domain_error("Only cosine similarity is currently implemented.");
        }
    }

    template <typename E>
    maelstrom::vector similarity_dispatch_emb(E exec_policy, similarity_t metric, maelstrom::vector& src_embeddings, maelstrom::vector& src_offsets, maelstrom::vector& target_embeddings, size_t emb_stride) {
        switch(src_embeddings.get_dtype().prim_type) {
            case UINT64:
                return launch_similarity<E, uint64_t>(exec_policy, metric, src_embeddings, src_offsets, target_embeddings, emb_stride);
            case UINT32:
                return launch_similarity<E, uint32_t>(exec_policy, metric, src_embeddings, src_offsets, target_embeddings, emb_stride);
            case UINT8:
                return launch_similarity<E, uint8_t>(exec_policy, metric, src_embeddings, src_offsets, target_embeddings, emb_stride);
            case INT64:
                return launch_similarity<E, int64_t>(exec_policy, metric, src_embeddings, src_offsets, target_embeddings, emb_stride);
            case INT32:
                return launch_similarity<E, int32_t>(exec_policy, metric, src_embeddings, src_offsets, target_embeddings, emb_stride);
            case INT8:
                return launch_similarity<E, int8_t>(exec_policy, metric, src_embeddings, src_offsets, target_embeddings, emb_stride);
            case FLOAT64:
                return launch_similarity<E, double>(exec_policy, metric, src_embeddings, src_offsets, target_embeddings, emb_stride);
            case FLOAT32:
                return launch_similarity<E, float>(exec_policy, metric, src_embeddings, src_offsets, target_embeddings, emb_stride);
        }

        throw std::runtime_error("Invalid data type encountered in similarity");
    }

    maelstrom::vector similarity(similarity_t metric, maelstrom::vector& src_embeddings, maelstrom::vector& src_offsets, maelstrom::vector& target_embeddings, size_t emb_stride) {
        if(maelstrom::is_dist(src_embeddings.get_mem_type())) {
            throw std::invalid_argument("The vector being compared cannot be distributed for similarity.");
        }
        
        if(maelstrom::is_dist(target_embeddings.get_mem_type())) {
            throw std::invalid_argument("The embeddings being compared to cannot be distributed for similarity.");
        }
        if(target_embeddings.get_dtype().prim_type != src_embeddings.get_dtype().prim_type) {
            throw std::invalid_argument("Primitive type of target embedding must match the source type for similarity.");
        }
        if(!src_offsets.empty() && src_offsets.get_dtype().prim_type != maelstrom::UINT64) {
            throw std::invalid_argument("Offsets must be of uint64 primitive type.");
        }
        

        std::any exec_policy = maelstrom::get_execution_policy(src_embeddings).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return similarity_dispatch_emb(
                std::any_cast<device_exec_t>(exec_policy),
                metric,
                src_embeddings,
                src_offsets,
                target_embeddings,
                emb_stride
            );
        } else if(typeid(host_exec_t) == t) {
            throw std::domain_error("The similarity algorithm is not supported for host execution.  Consider using pinned memory storage instead.");
        }

        throw std::runtime_error("Invalid execution policy for similarity");
    }

}