cmake_minimum_required(VERSION 3.10)

project(
    MAELSTROM
    VERSION 0.1
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_FLAGS --expt-relaxed-constexpr)

# set(CMAKE_CUDA_ARCHITECTURES "61;62;70;75;86")
set(CMAKE_CUDA_ARCHITECTURES "61")

set(POSITION_INDEPENDENT_CODE True)
set(INTERFACE_POSITION_INDEPENDENT_CODE True)

add_library(
    maelstrom

    # cpp
    containers/impl/common/vector_construct.cpp
    containers/impl/common/vector_modify.cpp
    containers/impl/common/vector_math.cpp
    storage/datatype.cpp
    algorithms/impl/common/unpack.cpp

    # cuda
    containers/impl/thrust/vector_malloc.cu
    containers/impl/thrust/vector_access.cu
    algorithms/impl/thrust/remove_if.cu
    algorithms/impl/thrust/increment.cu
    algorithms/impl/thrust/select.cu
    algorithms/impl/thrust/set.cu
    algorithms/impl/thrust/filter.cu
    algorithms/impl/thrust/compare.cu
    algorithms/impl/thrust/sort.cu
    algorithms/impl/thrust/unique.cu
    algorithms/impl/thrust/count_unique.cu
    algorithms/impl/thrust/reduce.cu
    algorithms/impl/thrust/reduce_by_key.cu
    algorithms/impl/thrust/math.cu
    algorithms/impl/thrust/cast.cu
    algorithms/impl/thrust/arange.cu
    algorithms/impl/thrust/intersection.cu
)

target_compile_options(
    maelstrom
    PUBLIC
        -O3
        -funsafe-math-optimizations
        -frename-registers
        -funroll-loops
        -fsized-deallocation
)

target_include_directories(
    maelstrom
    PUBLIC
        "/mnt/maelstrom/include/maelstrom" # maelstrom path
)

function(AddMaelstromTest EXEC_NAME)
    
    add_executable(${EXEC_NAME} ${ARGN})

    set_target_properties(
        ${EXEC_NAME}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "test/bin"
    )

    target_include_directories(
        ${EXEC_NAME}
        PRIVATE
            "/mnt/maelstrom/tests"
    )

    target_link_libraries(
        ${EXEC_NAME}
        PRIVATE
            maelstrom  
    )

endfunction()

AddMaelstromTest(test_vector.exe tests/test_vector.cpp)
AddMaelstromTest(test_remove.exe tests/test_remove.cpp)
AddMaelstromTest(test_increment.exe tests/test_increment.cpp)
AddMaelstromTest(test_set.exe tests/test_set.cpp)
AddMaelstromTest(test_select.exe tests/test_select.cpp)
AddMaelstromTest(test_unpack.exe tests/test_unpack.cpp)
AddMaelstromTest(test_filter.exe tests/test_filter.cpp)
AddMaelstromTest(test_compare.exe tests/test_compare.cpp)
AddMaelstromtest(test_sort.exe tests/test_sort.cpp)
AddMaelstromTest(test_unique.exe tests/test_unique.cpp)
AddMaelstromTest(test_count_unique.exe tests/test_count_unique.cpp)
AddMaelstromTest(test_reduce.exe tests/test_reduce.cpp)
AddMaelstromTest(test_reduce_by_key.exe tests/test_reduce_by_key.cpp)
AddMaelstromTest(test_math.exe tests/test_math.cpp)
AddMaelstromTest(test_cast.exe tests/test_cast.cpp)
AddMaelstromTest(test_arange.exe tests/test_arange.cpp)
AddMaelstromTest(test_intersection.exe tests/test_intersection.cpp)
