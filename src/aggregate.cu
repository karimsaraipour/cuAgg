#include "aggregate.cuh"

#include <cstdio>

__global__ void dummy_aggregate_kernel(const IndexT * const index,
    const NodeT * const neighbors, const FeatureT * const features,
    const NodeT num_nodes, const int num_features) {
  constexpr int PRINT_NODES_MAX = 2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Run everything from a single thread :)
  if (gid == 0) {
    // Print basic stats
    printf("Total nodes: %d\n", num_nodes);
    printf("Total features: %d\n", num_features);

    // Print nodes
    printf("The first %d nodes' edges are:\n", min(num_nodes, PRINT_NODES_MAX));
    for (NodeT n = 0; n < min(num_nodes, PRINT_NODES_MAX); n++)
      for (IndexT i = index[n]; i < index[n + 1]; i++)
        printf("%d -> %d\n", n, neighbors[i]);

    // Print feature vectors
    printf("The first %d nodes' feature vectors are:\n",
        min(num_nodes, PRINT_NODES_MAX));
    for (NodeT n = 0; n < min(num_nodes, PRINT_NODES_MAX); n++) {
      printf("Node %d: ", n);
      for (int f = 0; f < num_features; f++)
        printf("%d ", features[n * num_features + f]);
      printf("\n");
    }
  }
}
