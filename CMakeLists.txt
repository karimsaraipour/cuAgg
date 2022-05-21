cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(
  cuAgg
  VERSION 0.1
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set default architecture to target CC 7.0 (Tesla V100) and beyond
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

add_library(graph src/graph.cpp src/generator.cpp)
add_library(agg src/aggregate.cu)
set_target_properties(agg PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Utility: Graph generation
add_executable(generate_graph src/generate_graph.cpp)
target_link_libraries(generate_graph graph)

# Sample code: Generate graphs
add_executable(example_generate_graph src/example_generate_graph.cpp)
target_link_libraries(example_generate_graph graph)

# Sample code: Running CUDA kernels
add_executable(example_aggregate src/example_aggregate.cu)
target_link_libraries(example_aggregate graph agg)
