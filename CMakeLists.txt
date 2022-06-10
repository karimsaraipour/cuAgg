cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(
  cuAgg
  VERSION 0.1
  LANGUAGES CXX CUDA)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

option(WITH_TESTS "Generate test suite" ON)

# Global Configuration
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set default architecture to target CC 7.0 (Tesla V100) and beyond
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

# Build tests
if(WITH_TESTS)
  enable_testing()
endif()

# Optional: LibPNG
find_package(PNGwriter QUIET)

# External Libraries
add_subdirectory(deps/yaml-cpp) # yaml-cpp

# Internal Libraries
add_library(graph src/graph/graph.cpp src/graph/generator.cpp
                  src/graph/partition.cpp)
add_library(agg src/kernels/aggregate_cpu.cu src/kernels/aggregate.cu
                src/kernels/aggregate_db.cu)
add_library(testing tests/testing.cpp)
set_target_properties(agg PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Utility: Graph generation
add_executable(generate_graph utils/generate_graph.cpp)
target_link_libraries(generate_graph graph)

# Utility: Kernel profiling
add_executable(profile_kernels utils/profile_kernels.cu)
target_link_libraries(profile_kernels graph agg)

# Utility: Kernel profiling (sparsity)
add_executable(profile_kernels_sparsity utils/profile_kernels_sparsity.cu)
target_link_libraries(profile_kernels_sparsity graph agg yaml-cpp)

# Utility: Draw adjacency matrix
if(PNGwriter_FOUND)
  add_executable(visualize_graph utils/visualize_graph.cpp)
  target_link_libraries(visualize_graph graph PNGwriter::PNGwriter)
endif()

# Sample code: Generate graphs
add_executable(example_graph examples/example_graph.cpp)
target_link_libraries(example_graph graph)

# Sample code: Running CUDA kernels
add_executable(example_aggregate examples/example_aggregate.cu)
target_link_libraries(example_aggregate graph agg)

# Test: Check (de)serialization works
add_executable(test_generate_graph tests/test_generate_graph.cpp)
target_link_libraries(test_generate_graph testing graph agg)
add_test(NAME generate_graph COMMAND test_generate_graph)

# Test: Check square testing graph partitioning
add_executable(test_partition tests/test_partition.cpp)
target_link_libraries(test_partition testing graph agg)
add_test(NAME partition COMMAND test_partition)

# Test: Check parallel CPU aggregate
add_executable(test_aggregate_cpu tests/test_aggregate_cpu.cu)
target_link_libraries(test_aggregate_cpu testing graph agg)
add_test(NAME aggregate_cpu COMMAND test_aggregate_cpu)

# Test: Check GPU naive aggregate
add_executable(test_aggregate_naive tests/test_aggregate_naive.cu)
target_link_libraries(test_aggregate_naive testing graph agg)
add_test(NAME aggregate_naive COMMAND test_aggregate_naive)

# Test: Check GPU dyn aggregate
add_executable(test_aggregate_dyn tests/test_aggregate_dyn.cu)
target_link_libraries(test_aggregate_dyn testing graph agg)
add_test(NAME aggregate_dyn COMMAND test_aggregate_dyn)

# Test: Check GPU dyn aggregate shared memory
add_executable(test_aggregate_dyn_sm tests/test_aggregate_dyn_sm.cu)
target_link_libraries(test_aggregate_dyn_sm testing graph agg)
add_test(NAME aggregate_dyn_sm COMMAND test_aggregate_dyn_sm)

# Test: Check GPU dyn aggregate register file
add_executable(test_aggregate_dyn_rf tests/test_aggregate_dyn_rf.cu)
target_link_libraries(test_aggregate_dyn_rf testing graph agg)
add_test(NAME aggregate_dyn_rf COMMAND test_aggregate_dyn_rf)

# Test: Check GPU dyn aggregate register file + shared memory
add_executable(test_aggregate_dyn_sm_rf tests/test_aggregate_dyn_sm_rf.cu)
target_link_libraries(test_aggregate_dyn_sm_rf testing graph agg)
add_test(NAME aggregate_dyn_sm_rf COMMAND test_aggregate_dyn_sm_rf)

# Test: Check GPU partitioned aggregate
add_executable(test_aggregate_partition tests/test_aggregate_partition.cu)
target_link_libraries(test_aggregate_partition testing graph agg)
add_test(NAME aggregate_partition COMMAND test_aggregate_partition)

# Test: Check GPU double buffering aggregate
add_executable(test_aggregate_db_naive tests/test_aggregate_db_naive.cu)
target_link_libraries(test_aggregate_db_naive testing graph agg)
add_test(NAME aggregate_db_naive COMMAND test_aggregate_db_naive)

# Test: Check GPU double buffering aggregate (sliced)
add_executable(test_aggregate_db_naive_sliced
               tests/test_aggregate_db_naive_sliced.cu)
target_link_libraries(test_aggregate_db_naive_sliced testing graph agg)
add_test(NAME aggregate_db_naive_sliced COMMAND test_aggregate_db_naive_sliced)
