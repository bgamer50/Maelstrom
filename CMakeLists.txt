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
    containers/vector_construct.cpp
    containers/vector_modify.cpp
    storage/datatype.cpp

    # cuda
    containers/vector_malloc.cu
    containers/vector_access.cu
    algorithms/remove_if.cu
    algorithms/increment.cu
    algorithms/select.cu
    algorithms/set.cu
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
