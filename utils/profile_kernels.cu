#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "../src/cuda.cuh"
#include "../src/graph/generator.h"
#include "../src/graph/graph.h"
#include "../src/graph/partition.h"
#include "../src/kernels/aggregate.cuh"
#include "../src/kernels/aggregate_templated.cuh"

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

  // TODO: rm hack
  tile_size = 1024;
  sparsity = 1.0f;

  // Partition graph and report how many actually meet sparsity requirement
  auto partitions = partition_square_tile(g, tile_size);

  size_t w_tile_size = tile_size;
  size_t valid_ngh_size = w_tile_size * w_tile_size * sparsity;
  NodeT failed_tiles = 0;
  NodeT zero_tiles = 0;
  for (const auto &part : partitions) {
    const auto &subg = part.subgraph;
    IndexT num_edges = subg->index.get()[subg->num_idx_nodes];
    float tile_sparsity =
        static_cast<float>(num_edges) / (w_tile_size * w_tile_size);
    std::cout << num_edges << ' ' << tile_sparsity << std::endl;
    auto ngh_count = subg->index.get()[subg->num_idx_nodes];
    if (ngh_count > valid_ngh_size)
      failed_tiles++;
    if (ngh_count == 0)
      zero_tiles++;
  }

  // Print stats
  auto num_nodes = g->num_idx_nodes;
  auto num_edges = g->index.get()[g->num_idx_nodes];
  std::cout << "Graph" << std::endl
            << "  # of nodes: " << num_nodes << std::endl
            << "  # of edges: " << num_edges << std::endl
            << "  avg degree: " << (double)num_edges / num_nodes << std::endl;
  std::cout << std::endl;
  std::cout << "Tile" << std::endl
            << "  Tile size: " << tile_size << std::endl
            << "  Total tiles: " << partitions.size() << std::endl
            << "  Failed tiles: " << failed_tiles << std::endl
            << "  Zero tiles: " << zero_tiles << std::endl;

  // Generate profile
  auto features = generate_features(g->num_idx_nodes, num_features);
  size_t feature_count = g->num_idx_nodes * num_features;
  std::cout << "Features created" << std::endl;

  auto dummy_features = FeatureVector::create(feature_count);
  NodeT num_tiles1D = (g->num_idx_nodes + tile_size - 1) / tile_size;

  // Generate timing events
  cudaEvent_t *starts = new cudaEvent_t[partitions.size()];
  cudaEvent_t *stops = new cudaEvent_t[partitions.size()];
  for (size_t i = 0; i < partitions.size(); i++) {
    CUDA_ERRCHK(cudaEventCreate(&starts[i]));
    CUDA_ERRCHK(cudaEventCreate(&stops[i]));
  }
  IndexT e = 0;

  // Run kernel
  aggregate_double_buffer_naive(
      partitions, num_tiles1D, features, dummy_features, num_features,
      tile_size,
      [&starts, &stops,
       &e](const IndexT *const index, const NodeT *const neighbors,
           const FeatureT *const in_features, FeatureT *const out_features,
           const NodeT num_nodes, const IndexT num_features) -> void {
        cudaEventRecord(starts[e]);
        aggregate_dyn<<<num_nodes, 32>>>(index, neighbors, in_features,
                                         out_features, num_nodes, num_features);
        cudaEventRecord(stops[e]);
        e++;
      },
      db_size, valid_ngh_size);

  // Report times
  for (size_t i = 0; i < e; i++) {
    CUDA_ERRCHK(cudaEventSynchronize(stops[i]));
    float elapsed;
    CUDA_ERRCHK(cudaEventElapsedTime(&elapsed, starts[i], stops[i]));
    std::cout << "  Time: " << elapsed << " ms" << std::endl;
  }

  return EXIT_SUCCESS;
}
