#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <iterator>

#include "../src/graph/generator.h"
#include "../src/graph/graph.h"
#include "../src/graph/partition.h"

int main(int argc, char *argv[]) {
  constexpr int TEST_SCALE = 14;
  constexpr int TEST_DEGREE = 10;
  constexpr int TEST_TILE_SIZE = 16;

  auto g = generate_krongraph(TEST_SCALE, TEST_DEGREE);

  auto partitions = partition_square_tile(g, TEST_TILE_SIZE);

  NodeT num_nodes = g->num_idx_nodes;
  NodeT num_tiles1D = (num_nodes + TEST_TILE_SIZE - 1) / TEST_TILE_SIZE;

  IndexT num_edges = 0;
  for (NodeT ngh_tile = 0; ngh_tile < num_tiles1D; ngh_tile++) {
    NodeT num_idx_nodes = 0;
    for (NodeT idx_tile = 0; idx_tile < num_tiles1D; idx_tile++) {
      const auto &part = partitions[idx_tile * num_tiles1D + ngh_tile];

      // Track number of nodes in this column & total number of edges
      num_idx_nodes += part.subgraph->num_idx_nodes;
      num_edges += part.subgraph->index.get()[part.subgraph->num_idx_nodes];

      // Make sure tile is configured correctly
      assert(part.subgraph->direction == g->direction);
      assert(part.idx_map.type == NodeMapping::MappingT::affine);
      assert(part.ngh_map.type == NodeMapping::MappingT::affine);

      // Make sure the tile size is correct
      NodeT idx_start = idx_tile * TEST_TILE_SIZE;
      NodeT ngh_start = ngh_tile * TEST_TILE_SIZE;
      NodeT num_idx_nodes = std::min(num_nodes - idx_start, TEST_TILE_SIZE);
      NodeT num_ngh_nodes = std::min(num_nodes - ngh_start, TEST_TILE_SIZE);

      assert(part.subgraph->num_idx_nodes == num_idx_nodes);
      assert(part.subgraph->num_neighbors == num_ngh_nodes);
      assert(part.idx_map.base == idx_start);
      assert(part.ngh_map.base == ngh_start);
    }

    // Make sure the number of index nodes is correct
    assert(num_idx_nodes == num_nodes);
  }

  // Make sure the partition did not gain/lose any edges
  assert(num_edges == g->index.get()[g->num_idx_nodes]);

// Make sure edges are exist in the original graph
#define HAS_EDGE(g, u, v)                                                      \
  std::find(g->neighbors.get() + g->index.get()[v],                            \
            g->neighbors.get() + g->index.get()[v + 1],                        \
            u) != g->neighbors.get() + g->index.get()[v + 1]
#define MAP_IDX(part, v) part.idx_map.base + v
#define MAP_NGH(part, u) part.ngh_map.base + u

  for (const auto &part : partitions) {
    const auto &subgraph = part.subgraph;
    for (NodeT pv = 0; pv < subgraph->num_idx_nodes; pv++) {
      NodeT v = MAP_IDX(part, pv);
      for (IndexT i = subgraph->index.get()[pv];
           i < subgraph->index.get()[pv + 1]; i++) {
        NodeT pu = subgraph->neighbors.get()[i];
        NodeT u = MAP_NGH(part, pu);
        assert(HAS_EDGE(g, u, v));
      }
    }
  }

#undef MAP_IDX
#undef MAP_NGH
#undef HAS_EDGE

  return EXIT_SUCCESS;
}
