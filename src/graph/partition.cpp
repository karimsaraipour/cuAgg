#include "partition.h"

#include <algorithm>
#include <assert.h>

PartitionVec partition_square_tile(const GraphPtr g, const NodeT tile_size) {
  assert(g->direction == Graph::DirectionT::pull); // assume pull graph; can be
                                                   // extended to push easily
  assert(g->num_idx_nodes ==
         g->num_neighbors); // assume square adj matrix first

  // Number of tiles in one dimension.
  NodeT num_nodes = g->num_idx_nodes;
  NodeT num_tiles1D = (num_nodes + tile_size - 1) / tile_size;

  // Construct & initialize return vector
  PartitionVec partitions(num_tiles1D * num_tiles1D);
  for (NodeT idx_tile = 0; idx_tile < num_tiles1D; idx_tile++) {
    for (NodeT ngh_tile = 0; ngh_tile < num_tiles1D; ngh_tile++) {
      auto &part = partitions[idx_tile * num_tiles1D + ngh_tile];

      NodeT idx_start = idx_tile * tile_size;
      NodeT ngh_start = ngh_tile * tile_size;
      NodeT num_idx_nodes = std::min(num_nodes - idx_start, tile_size);
      NodeT num_ngh_nodes = std::min(num_nodes - ngh_start, tile_size);

      // Set subgraph direction
      part.subgraph->direction = g->direction;

      // Set graph size (matrix dimensions)
      part.subgraph->num_idx_nodes = num_idx_nodes;
      part.subgraph->num_neighbors = num_ngh_nodes;

      // Set arrays
      part.subgraph->index.resize(1);
      part.subgraph->index[0] = 0;
      part.subgraph->neighbors.resize(0);

      // Set mapping
      part.idx_map = NodeMapping::new_affine(idx_start);
      part.ngh_map = NodeMapping::new_affine(ngh_start);
    }
  }

  // Assign edges to the correct partition
  for (NodeT v = 0; v < g->num_idx_nodes; v++) {
    // Get appropriate src tile
    NodeT idx_tile = v / tile_size;

    // Push neighbors to appropriate neighbor list
    for (IndexT i = g->index[v]; i < g->index[v + 1]; i++) {
      NodeT u = g->neighbors[i];
      NodeT ngh_tile = u / tile_size;

      auto &part = partitions[idx_tile * num_tiles1D + ngh_tile];
      NodeT pu = u - ngh_tile * tile_size;
      part.subgraph->neighbors.push_back(pu);
    }

    // Update all subgraph indices this idx_tile touches
    for (NodeT ngh_tile = 0; ngh_tile < num_tiles1D; ngh_tile++) {
      auto &part = partitions[idx_tile * num_tiles1D + ngh_tile];
      part.subgraph->index.push_back(part.subgraph->neighbors.size());
    }
  }

  return partitions;
}
