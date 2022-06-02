#include <assert.h>
#include <cstdlib>
#include <utility>
#include <math.h>

#include "../src/aggregate.cuh"
#include "../src/generator.h"
#include "../src/graph.h"

bool feq(float f1, float f2) {
  return fabs(f1 - f2) < 0.001;
}


void aggregate_cpu_oracle(const GraphPtr g, const FeatureVec &in_features,
                          FeatureVec &out_features, int num_features) {
  FeatureVec node_features(num_features);

  for (NodeT v = 0; v < g->num_nodes; v++) {
    // Reset node features
    for (IndexT f = 0; f < num_features; f++)
      node_features[f] = in_features[v * num_features + f];

    // Aggregate features
    for (IndexT i = g->index[v]; i < g->index[v + 1]; i++) {
      NodeT u = g->neighbors[i];
      for (int f = 0; f < num_features; f++)
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
  constexpr IndexT TEST_NUM_FEATURES = 1024;

  // Generate graph
  auto g = generate_krongraph(TEST_SCALE, TEST_DEGREE);
  assert(g != nullptr && "graph is invalid");

  // Get CPU oracle (single-threaded)
  auto features = generate_features(g->num_nodes, TEST_NUM_FEATURES);
  assert(!features.empty() && "features are empty");
  FeatureVec oracle_features(features.size());
  FeatureVec test_features(features.size());

  aggregate_cpu_oracle(g, features, oracle_features, TEST_NUM_FEATURES);

  // Get CPU parallel aggregated features
  aggregate_cpu(g, features, test_features, TEST_NUM_FEATURES);

  for (size_t i = 0; i < features.size(); i++)
    assert(feq(test_features[i], oracle_features[i]) && "features don't match");

  return EXIT_SUCCESS;
}
