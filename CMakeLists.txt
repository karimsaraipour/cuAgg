cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(
  cuAgg
  VERSION 0.1
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(graph src/graph.cpp src/generator.cpp)

# Sample code: Generate graphs
add_executable(example_generate_graph src/example_generate_graph.cpp)
target_link_libraries(example_generate_graph graph)
