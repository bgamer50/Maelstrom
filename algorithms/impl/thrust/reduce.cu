#include "maelstrom/algorithms/reduce.h"
#include "maelstrom/thrust_utils/thrust_utils.cuh"
#include "maelstrom/thrust_utils/thrust_reduction.cuh"

namespace maelstrom {

    template<typename E, typename T>
    std::pair<boost::any, size_t> t_reduce(E exec_policy, maelstrom::vector& vec, maelstrom::reductor red) {
        auto zip = thrust::make_zip_iterator(
            thrust::make_tuple(
                maelstrom::device_tptr_cast<T>(vec.data()),
                thrust::make_counting_iterator<size_t>(static_cast<size_t>(0))
            )
        );

        switch(red) {
            case MIN: {
                auto t = thrust::reduce(
                    exec_policy,
                    zip,
                    zip + vec.size(),
                    thrust::make_tuple(static_cast<T>(0), std::numeric_limits<size_t>::max()),
                    red_min<T>()
                );

                return std::make_pair(
                    boost::any(thrust::get<0>(t)),
                    thrust::get<1>(t)
                );
            }
            case MAX: {
                auto t = thrust::reduce(
                    exec_policy,
                    zip,
                    zip + vec.size(),
                    thrust::make_tuple(static_cast<T>(0), std::numeric_limits<size_t>::max()),
                    red_max<T>()
                );

                return std::make_pair(
                    boost::any(thrust::get<0>(t)),
                    thrust::get<1>(t)
                );
            }
            case PRODUCT: {
                auto t = thrust::reduce(
                    exec_policy,
                    zip,
                    zip + vec.size(),
                    thrust::make_tuple(static_cast<T>(0), std::numeric_limits<size_t>::max()),
                    red_product<T>()
                );

                return std::make_pair(
                    boost::any(thrust::get<0>(t)),
                    thrust::get<1>(t)
                );
            }
            case SUM: {
                auto t = thrust::reduce(
                    exec_policy,
                    zip,
                    zip + vec.size(),
                    thrust::make_tuple(static_cast<T>(0), std::numeric_limits<size_t>::max()),
                    red_sum<T>()
                );

                return std::make_pair(
                    boost::any(thrust::get<0>(t)),
                    thrust::get<1>(t)
                );
            }
            case MEAN: {
                auto t = thrust::reduce(
                    exec_policy,
                    zip,
                    zip + vec.size(),
                    thrust::make_tuple(static_cast<T>(0), std::numeric_limits<size_t>::max()),
                    red_min<T>()
                );

                double v = static_cast<double>(thrust::get<0>(t));
                v /= static_cast<double>(vec.size());

                return std::make_pair(
                    v,
                    thrust::get<1>(t)
                );
            }
        }

        throw std::runtime_error("Invalid reductor provided to reduce");
    }

    template<typename E>
    std::pair<boost::any, size_t> reduce_dispatch_val(E exec_policy, maelstrom::vector& vec, maelstrom::reductor red) {
        switch(vec.get_dtype().prim_type) {
            case UINT64:
                return t_reduce<E, uint64_t>(exec_policy, vec, red);
            case UINT32:
                return t_reduce<E, uint32_t>(exec_policy, vec, red);
            case UINT8:
                return t_reduce<E, uint8_t>(exec_policy, vec, red);
            case INT64:
                return t_reduce<E, int64_t>(exec_policy, vec, red);
            case INT32:
                return t_reduce<E, int32_t>(exec_policy, vec, red);
            case INT8:
                return t_reduce<E, int8_t>(exec_policy, vec, red);
            case FLOAT64:
                return t_reduce<E, double>(exec_policy, vec, red);
            case FLOAT32:
                return t_reduce<E, float>(exec_policy, vec, red);
        }

        throw std::runtime_error("invalid primitive type provided to reduce");
    }

    std::pair<boost::any, size_t> reduce_dispatch_exec_policy(maelstrom::vector& vec, maelstrom::reductor red) {
        boost::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            return reduce_dispatch_val(
                boost::any_cast<device_exec_t>(exec_policy),
                vec,
                red
            );
        } else if(typeid(host_exec_t) == t) {
            return reduce_dispatch_val(
                boost::any_cast<host_exec_t>(exec_policy),
                vec,
                red
            );
        }

        throw std::runtime_error("Invalid execution policy for increment");
    }

    std::pair<boost::any, size_t> reduce(maelstrom::vector& vec, maelstrom::reductor red) {
        if(vec.size() == 0) {
            throw std::runtime_error("Attempting to reduce an empty vector!");
        }

        return reduce_dispatch_exec_policy(vec, red);
    }
    
}