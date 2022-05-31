#include "aggregate.cuh"

__global__ void
aggregate_naive(const IndexT *const index, const NodeT *const neighbors,
                const FeatureT *const in_features, FeatureT *const out_features,
                const NodeT num_nodes, const IndexT num_features) {
  NodeT start_v = blockIdx.x * blockDim.x + threadIdx.x;
  NodeT offset_v = blockDim.x * gridDim.x;
  IndexT start_f = blockIdx.y * blockDim.y + threadIdx.y;
  IndexT offset_f = blockDim.y * gridDim.y;

  for (NodeT v = start_v; v < num_nodes; v += offset_v) {
    // Write out output feature (itself)
    for (IndexT f = start_f; f < num_features; f += offset_f)
      out_features[v * num_features + f] = in_features[v * num_features + f];

    // Aggregate neighbors
    for (IndexT i = index[v]; i < index[v + 1]; i++) {
      NodeT u = neighbors[i];
      for (IndexT f = start_f; f < num_features; f += offset_f)
        out_features[v * num_features + f] += in_features[u * num_features + f];
    }
  }
}
