#ifndef SRC_GRAPH__PARTITION_H_
#define SRC_GRAPH__PARTITION_H_

#include <vector>

#include "graph.h"

struct Partition;
using PartitionVec = std::vector<Partition>;

/**
 * Describes different types of node mappings for subgraphs since CSR format
 * assumes that the source and destination nodes both start from 0.
 *
 * Since we're stuck with C++11, we can't use std::variant :(
 *
 * Let:
 *   1. nidO = original node id in the graph
 *   2. nidS = subgraph node id
 *
 * Supported types:
 *   1. affine - subgraph node id maps linearly starting from the base
 *               i.e., nidO = base + nidS
 *   2. custom - subgraph node id can map to any node id in the original graph
 *               i.e., nidO = node_map[nidS]
 */
struct NodeMapping {
  enum class MappingT { affine, custom } type;
  NodeT base;       // for affine mapping
  NodeVec node_map; // for custom mapping

  /**
   * Constructors
   * Use these constructors!! DON'T manually initialize.
   */

  static NodeMapping new_affine(NodeT base) { return {MappingT::affine, base}; }
  static NodeMapping new_custom(NodeVec &node_map) {
    return {MappingT::custom, 0, node_map};
  }
};

struct Partition {
  GraphPtr subgraph;
  NodeMapping idx_map;
  NodeMapping ngh_map;

  Partition() : subgraph(GraphPtr(new Graph())) {}
};

/**
 * Simple partitioning scheme that converts the adjacency matrix into square
 * tiles of tile_size length.
 *
 * Returns:
 *   A list of tiles starting from one destination tile and all of it's
 *   source tiles.
 *   e.g.,
 *     dst1src1, dst1src2, ... , dst1srcN,
 *     dst2src1, dst2src2, ... , dst2srcN,
 *     ... ,
 *     dstNsrc1, dstNsrc2, ..., dstNsrcN
 */
PartitionVec partition_square_tile(const GraphPtr g, const NodeT tile_size);

#endif // SRC_GRAPH__PARTITION_H_
