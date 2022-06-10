#include <cstdlib>
#include <fstream>
#include <iostream>

#include "../src/cuda.cuh"
#include "../src/graph/generator.h"
#include "../src/graph/graph.h"
#include "../src/kernels/aggregate.cuh"

constexpr int NUM_FEATURES = 16;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " [graph.g]" << std::endl;
    return EXIT_FAILURE;
  }

  // Load graph
  auto g = std::shared_ptr<Graph>(new Graph());
  std::ifstream ifs(argv[1], std::ifstream::in);
  ifs >> *g;

  // Generate feature vectors
  auto features = generate_features(g->num_idx_nodes, NUM_FEATURES);

  size_t feature_count = g->num_idx_nodes * NUM_FEATURES;

  // Copy data structures to GPU
  IndexT *cu_index;
  NodeT *cu_neighbors;
  FeatureT *cu_features;
  size_t size_index = (g->num_idx_nodes + 1) * sizeof(IndexT);
  size_t size_neighbors = (g->index.get()[g->num_idx_nodes]) * sizeof(NodeT);
  size_t size_features = feature_count * sizeof(FeatureT);
  CUDA_ERRCHK(cudaMalloc((void **)&cu_index, size_index));
  CUDA_ERRCHK(cudaMalloc((void **)&cu_neighbors, size_neighbors));
  CUDA_ERRCHK(cudaMalloc((void **)&cu_features, size_features));
  CUDA_ERRCHK(
      cudaMemcpy(cu_index, g->index.get(), size_index, cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cu_neighbors, g->neighbors.get(), size_neighbors,
                         cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cu_features, features.get(), size_features,
                         cudaMemcpyHostToDevice));

  /*dummy_aggregate_kernel<<<1, 32>>>(cu_index, cu_neighbors, cu_features,*/
  /*g->num_idx_nodes, NUM_FEATURES);*/

  // Free memory
  CUDA_ERRCHK(cudaFree(cu_index));
  CUDA_ERRCHK(cudaFree(cu_neighbors));
  CUDA_ERRCHK(cudaFree(cu_features));

  return EXIT_SUCCESS;
}
