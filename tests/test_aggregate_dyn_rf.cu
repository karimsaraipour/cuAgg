#include <assert.h>
#include <cstdlib>
#include <math.h>
#include <utility>

#include "../src/cuda.cuh"
#include "../src/graph/generator.h"
#include "../src/graph/graph.h"
#include "../src/kernels/aggregate_templated.cuh"

bool check(size_t i, float test, float oracle) {
  bool is_correct = fabs(test - oracle) < 0.01;
  if (!is_correct) {
    std::cerr << "Test failed at index " << i << std::endl;
    std::cerr << "CPU: " << test << std::endl;
    std::cerr << "GPU: " << oracle << std::endl;
    std::cerr << "Delta: " << fabs(test - oracle) << std::endl;
  }
  return is_correct;
}

void aggregate_cpu_oracle(const GraphPtr g, const FeatureVec &in_features,
                          FeatureVec &out_features, int num_features) {
  FeatureVec node_features(num_features);

  for (NodeT v = 0; v < g->num_idx_nodes; v++) {
    // Reset node features
    for (IndexT f = 0; f < num_features; f++)
      node_features[f] = 0;

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
  constexpr int TEST_SCALE = 14;
  constexpr int TEST_DEGREE = 10;
  constexpr IndexT TEST_NUM_FEATURES = 64;

  // Generate graph
  auto g = generate_krongraph(TEST_SCALE, TEST_DEGREE);
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

  auto kernel = aggregate_dyn_rf<TEST_NUM_FEATURES>;
  kernel<<<64, 32 * 32>>>(cu_index, cu_neighbors, cu_in_features,
                          cu_out_features, g->num_idx_nodes, TEST_NUM_FEATURES);

  // Copy results to CPU memory
  FeatureT *test_features = new FeatureT[features.size()];
  CUDA_ERRCHK(cudaMemcpy(test_features, cu_out_features, size_features,
                         cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < features.size(); i++)
    assert(check(i, test_features[i], oracle_features[i]) &&
           "features don't match");

  delete[] test_features;

  return EXIT_SUCCESS;
}
