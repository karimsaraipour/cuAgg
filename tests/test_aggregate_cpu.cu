#include <assert.h>
#include <cstdlib>
#include <math.h>
#include <utility>

#include "../src/graph/generator.h"
#include "../src/graph/graph.h"
#include "../src/kernels/aggregate.cuh"
#include "testing.h"

int main() {
  constexpr int TEST_SCALE = 14;
  constexpr int TEST_DEGREE = 10;
  constexpr IndexT TEST_NUM_FEATURES = 1024;

  // Generate graph
  auto g = generate_krongraph(TEST_SCALE, TEST_DEGREE);
  assert(g != nullptr && "graph is invalid");

  // Get CPU oracle (single-threaded)
  auto features = generate_features(g->num_idx_nodes, TEST_NUM_FEATURES);

  size_t feature_count = g->num_idx_nodes * TEST_NUM_FEATURES;
  auto oracle_features = FeatureVector::create(feature_count);
  auto test_features = FeatureVector::create(feature_count);

  aggregate_cpu_oracle(g, features, oracle_features, TEST_NUM_FEATURES);

  // Get CPU parallel aggregated features
  aggregate_cpu(g, features, test_features, TEST_NUM_FEATURES);

  for (size_t i = 0; i < feature_count; i++)
    assert(check(i, test_features.get()[i], oracle_features.get()[i]) &&
           "features don't match");

  return EXIT_SUCCESS;
}
