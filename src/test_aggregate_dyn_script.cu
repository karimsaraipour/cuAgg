#include <assert.h>
#include <cstdlib>
#include <utility>
#include <math.h>
#include <chrono>

#include "./kernels/aggregate.cuh"
#include "cuda.cuh"
#include "./graph/generator.h"
#include "./graph/graph.h"

#ifndef SCALE
#define SCALE 10
#endif

#ifndef DEGREE
#define DEGREE 50
#endif

#ifndef WARP
#define WARP 4
#endif

#ifndef THRESHOLD_LOW
#define THRESHOLD_LOW 8
#endif

#ifndef THRESHOLD_MED
#define THRESHOLD_MED 20
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

bool feq(float f1, float f2) {
  return fabs(f1 - f2) < 0.001;
}

void aggregate_cpu_oracle(const GraphPtr g, const FeatureVec &in_features,
                          FeatureVec &out_features, int num_features) {
  FeatureVec node_features(num_features);

  for (NodeT v = 0; v < g->num_idx_nodes; v++) {
    // Reset node features
    for (IndexT f = 0; f < num_features; f++)
      node_features[f] = in_features[v * num_features + f];

    // Aggregate features
    for (IndexT i = g->index[v]; i < g->index[v + 1]; i++) {
      NodeT u = g->neighbors[i];
      for (IndexT f = 0; f < num_features; f++)
        node_features[f] += in_features[u * num_features + f];
    }

    // Write out
    for (IndexT f = 0; f < num_features; f++)
      out_features[v * num_features + f] = node_features[f];
  }
}

int main() {
  // constexpr int TEST_SCALE = 14;
  // constexpr int TEST_DEGREE = 10;
  constexpr IndexT TEST_NUM_FEATURES = 64;

  constexpr int BLOCK_DIM_X = 16;
  constexpr int BLOCK_DIM_Y = 32;

  // Generate graph
  auto g = generate_krongraph(SCALE, DEGREE);
  assert(g != nullptr && "graph is invalid");

  // Get CPU oracle (single-threaded)
  auto features = generate_features(g->num_idx_nodes, TEST_NUM_FEATURES);
  assert(!features.empty() && "features are empty");
  FeatureVec oracle_features(features.size());

  aggregate_cpu_oracle(g, features, oracle_features, TEST_NUM_FEATURES);

  // Get GPU aggregated features
  IndexT *cu_index;
  NodeT *cu_neighbors;
  FeatureT *cu_in_features;
  FeatureT *cu_out_features;
  size_t size_index = g->index.size() * sizeof(IndexT);
  size_t size_neighbors = g->neighbors.size() * sizeof(NodeT);
  size_t size_features = features.size() * sizeof(FeatureT);
  CUDA_ERRCHK(cudaMalloc((void **)&cu_index, size_index));
  CUDA_ERRCHK(cudaMalloc((void **)&cu_neighbors, size_neighbors));
  CUDA_ERRCHK(cudaMalloc((void **)&cu_in_features, size_features));
  CUDA_ERRCHK(cudaMalloc((void **)&cu_out_features, size_features));
  CUDA_ERRCHK(cudaMemcpy(cu_index, g->index.data(), size_index,
                         cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cu_neighbors, g->neighbors.data(), size_neighbors,
                         cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cu_in_features, features.data(), size_features,
                         cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemset(cu_out_features, 0, size_features));

  dim3 dim_block(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 dim_grid((g->num_idx_nodes + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                (TEST_NUM_FEATURES + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

  size_t num_warps_pl;
  
  if (g->num_neighbors < THRESHOLD_LOW)
    num_warps_pl = 1;
  else if(g->num_neighbors < THRESHOLD_MED && g->num_neighbors > THRESHOLD_LOW)
    num_warps_pl = 4;
  else
    num_warps_pl = 10;

  auto start = std::chrono::high_resolution_clock::now();
  aggregate_dyn<<<64, num_warps_pl * WARP_SIZE, TEST_NUM_FEATURES * sizeof(FeatureT)>>>(cu_index, cu_neighbors,
                                     cu_in_features, cu_out_features,
                                     g->num_idx_nodes, TEST_NUM_FEATURES);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  auto kernel_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e9;
  fprintf(stdout, "Kernel execution time for %d warps: %f s\n", num_warps_pl, kernel_time);

  // Copy results to CPU memory
  FeatureT *test_features = new FeatureT[features.size()];
  CUDA_ERRCHK(cudaMemcpy(test_features, cu_out_features, size_features,
                         cudaMemcpyDeviceToHost));

  // for (size_t i = 0; i < features.size(); i++)
  //   assert(feq(test_features[i], oracle_features[i]) && "features don't match");

  delete[] test_features;

  return EXIT_SUCCESS;
}
