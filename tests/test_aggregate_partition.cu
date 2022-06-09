#include <assert.h>
#include <cstdlib>
#include <math.h>
#include <utility>

#include "../src/cuda.cuh"
#include "../src/graph/generator.h"
#include "../src/graph/graph.h"
#include "../src/graph/partition.h"
#include "../src/kernels/aggregate.cuh"

bool check(size_t i, float test, float oracle) {
  bool is_correct = fabs(test - oracle) < 0.001;
  if (!is_correct) {
    std::cerr << "Test failed at index " << i << std::endl;
    std::cerr << "CPU: " << test << std::endl;
    std::cerr << "GPU: " << oracle << std::endl;
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
  constexpr NodeT TEST_TILE_SIZE = 1 << 9;

  // Generate graph
  auto g = generate_krongraph(TEST_SCALE, TEST_DEGREE);
  assert(g != nullptr && "graph is invalid");

  // Get CPU oracle (single-threaded)
  auto features = generate_features(g->num_idx_nodes, TEST_NUM_FEATURES);
  assert(!features.empty() && "features are empty");
  FeatureVec oracle_features(features.size());

  aggregate_cpu_oracle(g, features, oracle_features, TEST_NUM_FEATURES);

  // Copy results to CPU memory
  FeatureT *test_features = new FeatureT[features.size()];

  // Create partitions
  NodeT num_tiles1D = (g->num_idx_nodes + TEST_TILE_SIZE - 1) / TEST_TILE_SIZE;
  auto partitions = partition_square_tile(g, TEST_TILE_SIZE);

  aggregate_double_buffer_naive(
      partitions, num_tiles1D, features, test_features, TEST_NUM_FEATURES,
      TEST_TILE_SIZE,
      [](const IndexT *const index, const NodeT *const neighbors,
         const FeatureT *const in_features, FeatureT *const out_features,
         const NodeT num_nodes, const IndexT num_features) -> void {
        aggregate_dyn<<<num_nodes, 32>>>(index, neighbors, in_features,
                                         out_features, num_nodes, num_features);
      },
      1);

  for (size_t i = 0; i < features.size(); i++)
    assert(check(i, test_features[i], oracle_features[i]) &&
           "features don't match");

  delete[] test_features;

  return EXIT_SUCCESS;
}
