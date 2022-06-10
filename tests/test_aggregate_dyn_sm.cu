#include <assert.h>
#include <cstdlib>
#include <math.h>
#include <utility>

#include "../src/cuda.cuh"
#include "../src/graph/generator.h"
#include "../src/graph/graph.h"
#include "../src/kernels/aggregate.cuh"
#include "../src/kernels/aggregate_templated.cuh"
#include "testing.h"

int main() {
  constexpr int TEST_SCALE = 14;
  constexpr int TEST_DEGREE = 10;
  constexpr IndexT TEST_NUM_FEATURES = 64;

  // Generate graph
  auto g = generate_krongraph(TEST_SCALE, TEST_DEGREE);
  assert(g != nullptr && "graph is invalid");

  // Get CPU oracle (single-threaded)
  auto features = generate_features(g->num_idx_nodes, TEST_NUM_FEATURES);

  size_t feature_count = g->num_idx_nodes * TEST_NUM_FEATURES;
  auto oracle_features = FeatureVector::create(feature_count);

  aggregate_cpu_oracle(g, features, oracle_features, TEST_NUM_FEATURES);

  // Get GPU aggregated features
  IndexT *cu_index;
  NodeT *cu_neighbors;
  FeatureT *cu_in_features;
  FeatureT *cu_out_features;
  size_t size_index = (g->num_idx_nodes + 1) * sizeof(IndexT);
  size_t size_neighbors = (g->index.get()[g->num_idx_nodes]) * sizeof(NodeT);
  size_t size_features = feature_count * sizeof(FeatureT);
  CUDA_ERRCHK(cudaMalloc((void **)&cu_index, size_index));
  CUDA_ERRCHK(cudaMalloc((void **)&cu_neighbors, size_neighbors));
  CUDA_ERRCHK(cudaMalloc((void **)&cu_in_features, size_features));
  CUDA_ERRCHK(cudaMalloc((void **)&cu_out_features, size_features));
  CUDA_ERRCHK(
      cudaMemcpy(cu_index, g->index.get(), size_index, cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cu_neighbors, g->neighbors.get(), size_neighbors,
                         cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cu_in_features, features.get(), size_features,
                         cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemset(cu_out_features, 0, size_features));

  auto kernel = aggregate_dyn_sm<TEST_NUM_FEATURES>;
  kernel<<<64, 32 * 32>>>(cu_index, cu_neighbors, cu_in_features,
                          cu_out_features, g->num_idx_nodes, TEST_NUM_FEATURES);

  // Copy results to CPU memory
  auto test_features = FeatureVector::create(feature_count);
  CUDA_ERRCHK(cudaMemcpy(test_features.get(), cu_out_features, size_features,
                         cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < feature_count; i++)
    assert(check(i, test_features.get()[i], oracle_features.get()[i]) &&
           "features don't match");

  return EXIT_SUCCESS;
}
