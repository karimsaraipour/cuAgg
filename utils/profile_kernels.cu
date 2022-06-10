#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "../src/graph/graph.h"
#include "../src/graph/partition.h"
#include "../src/kernels/aggregate.cuh"

int main(int argc, char *argv[]) {
  if (argc < 3 && argc > 5) {
    std::cerr << "Usage: " << argv[0]
              << " <graph_file> <num_features> [<num_buffers> [<sparsity>]]"
              << std::endl;
    return EXIT_FAILURE;
  }

  // Load graph
  auto g = GraphPtr(new Graph());
  std::ifstream ifs(argv[1], std::ifstream::in);
  ifs >> *g;
  ifs.close();

  // Compute tile size
  IndexT num_features = atoi(argv[2]);
  int db_size = (argc >= 4) ? atoi(argv[3]) : 2;
  float sparsity = (argc >= 5) ? atof(argv[4]) : 0.2f;

  NodeT tile_size = std::min(
      get_square_tile_size(num_features, db_size, sparsity), g->num_idx_nodes);

  // Partition graph and report how many actually meet sparsity requirement
  auto partitions = partition_square_tile(g, tile_size);

  size_t w_tile_size = tile_size;
  size_t valid_ngh_size = w_tile_size * w_tile_size * sparsity;
  std::cout << valid_ngh_size << std::endl;
  NodeT failed_tiles = 0;
  for (auto &part : partitions) {
    IndexT num_edges = part.subgraph->index[part.subgraph->num_idx_nodes];
    float tile_sparsity =
        static_cast<float>(num_edges) / (w_tile_size * w_tile_size);
    std::cout << num_edges << ' ' << tile_sparsity << std::endl;
    if (part.subgraph->neighbors.size() > valid_ngh_size)
      failed_tiles++;
  }

  auto num_nodes = g->num_idx_nodes;
  auto num_edges = g->index[g->num_idx_nodes];
  std::cout << "Graph" << std::endl
            << "  # of nodes: " << num_nodes << std::endl
            << "  # of edges: " << num_edges << std::endl
            << "  avg degree: " << (double)num_edges / num_nodes << std::endl;
  std::cout << std::endl;
  std::cout << "Tile" << std::endl
            << "  Tile size: " << tile_size << std::endl
            << "  Total tiles: " << partitions.size() << std::endl
            << "  Failed tiles: " << failed_tiles << std::endl;

  return EXIT_SUCCESS;
}
