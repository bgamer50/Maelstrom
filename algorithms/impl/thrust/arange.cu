#include "maelstrom/algorithms/arange.h"
#include "maelstrom/thrust_utils/thrust_utils.cuh"
#include "maelstrom/thrust_utils/execution.cuh"

namespace maelstrom {

    template <typename E, typename T>
    void t_fill_arange(E exec_policy, maelstrom::vector& vec, std::any start, std::any end, std::any inc) {
        T t_start = std::any_cast<T>(start);
        T t_end = std::any_cast<T>(end);
        T t_inc = std::any_cast<T>(inc);

        if(t_inc <= 0) throw std::runtime_error("increment must be > 0 for arange!");
        if(t_start >= t_end) throw std::runtime_error("start must be < end for arange!");

        size_t num_values = static_cast<size_t>((t_end - t_start) / t_inc) + (((t_end - t_start) % t_inc) > 0 ? 1 : 0);

        vec.resize(num_values);

        maelstrom::unary_times_op<T> times_op;
        times_op.times_val = t_inc;

        maelstrom::unary_plus_op<T> plus_op;
        plus_op.plus_val = t_start;
        
        auto counter = thrust::make_counting_iterator(static_cast<T>(0));
        auto multiplier = thrust::make_transform_iterator(counter, times_op);
        auto summer = thrust::make_transform_iterator(multiplier, plus_op);

        thrust::copy(
            exec_policy,
            summer,
            summer + num_values,
            maelstrom::device_tptr_cast<T>(vec.data())
        );
    }

    template<typename T>
    maelstrom::vector arange_dispatch_exec_policy(maelstrom::storage mem_type, std::any start, std::any end, std::any inc) {
        maelstrom::vector vec(
            mem_type,
            maelstrom::dtype_from_prim_type(maelstrom::prim_type_of(start))
        );

        std::any exec_policy = maelstrom::get_execution_policy(vec).get();
        const std::type_info& t = exec_policy.type();
        
        if(typeid(device_exec_t) == t) {
            t_fill_arange<device_exec_t, T>(
                std::any_cast<device_exec_t>(exec_policy),
                vec,
                start,
                end,
                inc
            );
        } else if(typeid(host_exec_t) == t) {
             t_fill_arange<host_exec_t, T>(
                std::any_cast<host_exec_t>(exec_policy),
                vec,
                start,
                end,
                inc
            );
        } else {
            throw std::runtime_error("Invalid execution policy");
        }

        return vec;
    }

    maelstrom::vector arange(maelstrom::storage mem_type, std::any start, std::any end, std::any inc) {
        if(start.type() != end.type()) throw std::runtime_error("Start and end dtype must match in arange!");
        if(start.type() != inc.type()) throw std::runtime_error("Start dtype must match increment dtype in arange!");

        if(start.type() == typeid(uint64_t)) return arange_dispatch_exec_policy<uint64_t>(mem_type, start, end, inc);
        if(start.type() == typeid(uint32_t)) return arange_dispatch_exec_policy<uint32_t>(mem_type, start, end, inc);
        if(start.type() == typeid(uint8_t)) return arange_dispatch_exec_policy<uint8_t>(mem_type, start, end, inc);
        if(start.type() == typeid(int64_t)) return arange_dispatch_exec_policy<int64_t>(mem_type, start, end, inc);
        if(start.type() == typeid(int32_t)) return arange_dispatch_exec_policy<int32_t>(mem_type, start, end, inc);
        if(start.type() == typeid(int8_t)) return arange_dispatch_exec_policy<int8_t>(mem_type, start, end, inc);
        if(start.type() == typeid(double)) throw std::runtime_error("float64 not permitted for arange");
        if(start.type() == typeid(float)) throw std::runtime_error("float32 not permitted for arange");

        throw std::runtime_error("Unsupported dtype for arange");
        
    }

    template<typename T>
    maelstrom::vector arange_start_end_helper(maelstrom::storage mem_type, std::any start, std::any end) {
        return arange(mem_type, start, end, static_cast<T>(1));
    }

    maelstrom::vector arange(maelstrom::storage mem_type, std::any start, std::any end) {
        if(start.type() != end.type()) throw std::runtime_error("Start and end dtype must match in arange!");

        if(start.type() == typeid(uint64_t)) return arange_start_end_helper<uint64_t>(mem_type, start, end);
        if(start.type() == typeid(uint32_t)) return arange_start_end_helper<uint32_t>(mem_type, start, end);
        if(start.type() == typeid(uint8_t)) return arange_start_end_helper<uint8_t>(mem_type, start, end);
        if(start.type() == typeid(int64_t)) return arange_start_end_helper<int64_t>(mem_type, start, end);
        if(start.type() == typeid(int32_t)) return arange_start_end_helper<int32_t>(mem_type, start, end);
        if(start.type() == typeid(int8_t)) return arange_start_end_helper<int8_t>(mem_type, start, end);
        if(start.type() == typeid(double)) throw std::runtime_error("float64 not permitted for arange");
        if(start.type() == typeid(float)) throw std::runtime_error("float32 not permitted for arange");

        throw std::runtime_error("Unsupported dtype for arange");
    }


    template <typename T>
    maelstrom::vector arange_N_helper(maelstrom::storage mem_type, std::any N) {
        return arange(mem_type, static_cast<T>(0), N);
    }

    maelstrom::vector arange(maelstrom::storage mem_type, std::any N) {
        if(N.type() == typeid(uint64_t)) return arange_N_helper<uint64_t>(mem_type, N);
        if(N.type() == typeid(uint32_t)) return arange_N_helper<uint32_t>(mem_type, N);
        if(N.type() == typeid(uint8_t)) return arange_N_helper<uint8_t>(mem_type, N);
        if(N.type() == typeid(int64_t)) return arange_N_helper<int64_t>(mem_type, N);
        if(N.type() == typeid(int32_t)) return arange_N_helper<int32_t>(mem_type, N);
        if(N.type() == typeid(int8_t)) return arange_N_helper<int8_t>(mem_type, N);
        if(N.type() == typeid(double)) throw std::runtime_error("float64 not permitted for arange");
        if(N.type() == typeid(float)) throw std::runtime_error("float32 not permitted for arange");

        throw std::runtime_error("Unsupported dtype for arange");
    }

}