#include "aggregate.cuh"

#include <algorithm>

void aggregate_cpu(const GraphPtr g, const FeatureVec &in_features,
                   FeatureVec &out_features, IndexT num_features) {
  // Local copy of aggregated features for a node
  FeatureVec node_features(num_features);

#pragma omp parallel for private(node_features)
  for (NodeT v = 0; v < g->num_nodes; v++) {
    // Node features start with itself
    for (IndexT f = 0; f < num_features; f++)
      node_features[f] = in_features[v * num_features + f];

    // Aggregate negihbors
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
