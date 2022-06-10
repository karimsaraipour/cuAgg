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

      // Generate index array
      part.subgraph->index = IndexVector::create(num_idx_nodes + 1);
      for (IndexT idx = 0; idx < num_idx_nodes + 1; idx++)
        part.subgraph->index.get()[idx] = 0;

      // Set mapping
      part.idx_map = NodeMapping::new_affine(idx_start);
      part.ngh_map = NodeMapping::new_affine(ngh_start);
    }
  }

  // Dry run to determine the index array
#pragma omp parallel for
  for (NodeT idx = 0; idx < g->num_idx_nodes; idx++) {
    NodeT idx_tile = idx / tile_size;
    NodeT pidx = idx - idx_tile * tile_size;

    for (IndexT i = g->index.get()[idx]; i < g->index.get()[idx + 1]; i++) {
      NodeT ngh = g->neighbors.get()[i];
      NodeT ngh_tile = ngh / tile_size;

      auto &subg = partitions[idx_tile * num_tiles1D + ngh_tile].subgraph;

      // Update index array
#pragma omp atomic
      subg->index.get()[pidx + 1]++;
    }
  }

  // Prefix sum
#pragma omp parallel for
  for (auto i = 0; i < partitions.size(); i++) {
    auto subg = partitions[i].subgraph;
    for (NodeT i = 0; i < subg->num_idx_nodes; i++)
      subg->index.get()[i + 1] += subg->index.get()[i];
  }

  // Generate edge array
  for (auto &part : partitions)
    part.subgraph->neighbors = NodeVector::create(
        part.subgraph->index.get()[part.subgraph->num_idx_nodes]);

    // Assign edges to the correct partition
#pragma omp parallel for
  for (NodeT idx = 0; idx < g->num_idx_nodes; idx++) {
    // Get appropriate src tile
    NodeT idx_tile = idx / tile_size;
    NodeT pidx = idx - idx_tile * tile_size;

    // Push neighbors to appropriate neighbor list
    std::vector<IndexT> offsets(num_tiles1D, 0);
    for (IndexT i = g->index.get()[idx]; i < g->index.get()[idx + 1]; i++) {
      NodeT ngh = g->neighbors.get()[i];
      NodeT ngh_tile = ngh / tile_size;

      auto &subg = partitions[idx_tile * num_tiles1D + ngh_tile].subgraph;
      NodeT pngh = ngh - ngh_tile * tile_size;

      subg->neighbors.get()[subg->index.get()[pidx] + offsets[ngh_tile]] = pngh;
      offsets[ngh_tile]++;
    }
  }

  return partitions;
}
