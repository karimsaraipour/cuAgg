#include "aggregate.cuh"

#include <algorithm>

void aggregate_cpu(const GraphPtr g, const FeatureVec &features,
                   FeatureVec &out_features, int num_features) {
  // Local copy of aggregated features for a node
  FeatureVec node_features(num_features);

#pragma omp parallel for private(node_features)
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