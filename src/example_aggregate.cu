#include <cstdlib>
#include <fstream>
#include <iostream>

#include "aggregate.cuh"
#include "cuda.cuh"
#include "generator.h"
#include "graph.h"

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
  auto features = generate_features(g->num_nodes, NUM_FEATURES);

  // Copy data structures to GPU
  IndexT *cu_index;
  NodeT *cu_neighbors;
  FeatureT *cu_features;
  size_t index_size = g->index.size() * sizeof(IndexT);
  size_t neighbors_size = g->neighbors.size() * sizeof(NodeT);
  size_t features_size = features.size() * sizeof(FeatureT);
  CUDA_ERRCHK(cudaMalloc((void **)&cu_index, index_size));
  CUDA_ERRCHK(cudaMalloc((void **)&cu_neighbors, neighbors_size));
  CUDA_ERRCHK(cudaMalloc((void **)&cu_features, features_size));
  CUDA_ERRCHK(cudaMemcpy(cu_index, g->index.data(), index_size,
                         cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cu_neighbors, g->neighbors.data(), neighbors_size,
                         cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cu_features, features.data(), features_size,
                         cudaMemcpyHostToDevice));

  dummy_aggregate_kernel<<<1, 32>>>(cu_index, cu_neighbors, cu_features,
                                    g->num_nodes, NUM_FEATURES);

  // Free memory
  CUDA_ERRCHK(cudaFree(cu_index));
  CUDA_ERRCHK(cudaFree(cu_neighbors));
  CUDA_ERRCHK(cudaFree(cu_features));

  return EXIT_SUCCESS;
}
