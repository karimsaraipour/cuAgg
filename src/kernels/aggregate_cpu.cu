#include "aggregate.cuh"

#include <algorithm>

void aggregate_cpu_oracle(const GraphPtr g, const FeatureVec &in_features,
                          FeatureVec &out_features, int num_features) {
  FeatureVec node_features = FeatureVector::create(num_features);

  for (NodeT v = 0; v < g->num_idx_nodes; v++) {
    // Reset node features
    for (IndexT f = 0; f < num_features; f++)
      node_features.get()[f] = 0;

    // Aggregate features
    for (IndexT i = g->index.get()[v]; i < g->index.get()[v + 1]; i++) {
      NodeT u = g->neighbors.get()[i];
      for (int f = 0; f < num_features; f++)
        node_features.get()[f] += in_features.get()[u * num_features + f];
    }

    // Write out
    for (IndexT f = 0; f < num_features; f++)
      out_features.get()[v * num_features + f] = node_features.get()[f];
  }
}

void aggregate_cpu(const GraphPtr g, const FeatureVec &in_features,
                   FeatureVec &out_features, IndexT num_features) {
  // Local copy of aggregated features for a node
  FeatureVec node_features = FeatureVector::create(num_features);

#pragma omp parallel for private(node_features)
  for (NodeT v = 0; v < g->num_idx_nodes; v++) {
    // Node features start with itself
    for (IndexT f = 0; f < num_features; f++)
      node_features.get()[f] = 0;

    // Aggregate negihbors
    for (IndexT i = g->index.get()[v]; i < g->index.get()[v + 1]; i++) {
      NodeT u = g->neighbors.get()[i];
      for (IndexT f = 0; f < num_features; f++)
        node_features.get()[f] += in_features.get()[u * num_features + f];
    }

    // Write out
    for (IndexT f = 0; f < num_features; f++)
      out_features.get()[v * num_features + f] = node_features.get()[f];
  }
}
