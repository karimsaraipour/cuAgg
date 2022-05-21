#include <assert.h>
#include <utility>

#include "../src/aggregate.cuh"
#include "../src/generator.h"
#include "../src/graph.h"

void aggregate_cpu_oracle(const GraphPtr g, const FeatureVec &features,
                          FeatureVec &out_features, int num_features) {
  FeatureVec node_features(num_features);

  for (NodeT v = 0; v < g->num_nodes; v++) {
    // Reset node features
    std::fill(node_features.begin(), node_features.end(), 0);

    // Aggregate features
    for (IndexT i = g->index[v]; i < g->index[v + 1]; i++) {
      NodeT u = g->neighbors[i];
      for (int f = 0; f < num_features; f++)
        node_features[f] += features[u * num_features + f];
    }

    // Write out
    for (int f = 0; f < num_features; f++)
      out_features[v * num_features + f] = node_features[f];
  }
}

int main() {
  constexpr int test_scale = 10;
  constexpr int test_degree = 8;
  constexpr int test_num_features = 32;

  auto g = generate_krongraph(test_scale, test_degree);
  assert(g != nullptr && "graph is invalid");

  auto features = generate_features(g->num_nodes, test_num_features);
  assert(!features.empty() && "features are empty");
  FeatureVec oracle_features(features.size());
  FeatureVec test_features(features.size());

  aggregate_cpu_oracle(g, features, oracle_features, test_num_features);
  aggregate_cpu(g, features, test_features, test_num_features);

  for (size_t i = 0; i < features.size(); i++)
    assert(test_features[i] == oracle_features[i] && "features don't match");
}
