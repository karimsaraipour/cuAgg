#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <math.h>
#include <utility>

#include "../src/cuda.cuh"
#include "../src/graph/generator.h"
#include "../src/graph/graph.h"
#include "../src/graph/partition.h"
#include "../src/kernels/aggregate.cuh"
#include "testing.h"

int main() {
  constexpr int TEST_SCALE = 14;
  constexpr int TEST_DEGREE = 10;
  constexpr IndexT TEST_NUM_FEATURES = 64;
  constexpr NodeT TEST_TILE_SIZE = 1 << 9;

  // Generate graph
  auto g = generate_krongraph(TEST_SCALE, TEST_DEGREE);
  assert(g != nullptr && "graph is invalid");

  // Get CPU oracle (single-threaded)
  auto features = generate_features(g->num_idx_nodes, TEST_NUM_FEATURES);

  size_t feature_count = g->num_idx_nodes * TEST_NUM_FEATURES;
  auto oracle_features = FeatureVector::create(feature_count);

  aggregate_cpu_oracle(g, features, oracle_features, TEST_NUM_FEATURES);

  // Copy results to CPU memory
  auto test_features = FeatureVector::create(feature_count);

  // Create partitions
  NodeT num_tiles1D = (g->num_idx_nodes + TEST_TILE_SIZE - 1) / TEST_TILE_SIZE;
  auto partitions = partition_square_tile(g, TEST_TILE_SIZE);

  // Create subpartition
  int part_idx_offset = num_tiles1D / 3;
  int part_idx_cols = num_tiles1D / 4;
  auto part_slice = PartitionVec(
      partitions.begin() + part_idx_offset * num_tiles1D,
      partitions.begin() + (part_idx_offset + part_idx_cols) * num_tiles1D);

  aggregate_double_buffer_naive(
      part_slice, part_idx_cols, features, test_features, TEST_NUM_FEATURES,
      TEST_TILE_SIZE,
      [](const IndexT *const index, const NodeT *const neighbors,
         const FeatureT *const in_features, FeatureT *const out_features,
         const NodeT num_nodes, const IndexT num_features) -> void {
        aggregate_dyn<<<num_nodes, 32>>>(index, neighbors, in_features,
                                         out_features, num_nodes, num_features);
      });

  // Check feature vectors in subpartition
  NodeT idx_start = part_slice.begin()->idx_map.base;
  auto part_last = part_slice.end() - 1;
  NodeT idx_end = part_last->idx_map.base + part_last->subgraph->num_idx_nodes;
  for (size_t i = idx_start * TEST_NUM_FEATURES;
       i < idx_end * TEST_NUM_FEATURES; i++)
    assert(check(i, test_features.get()[i], oracle_features.get()[i]) &&
           "features don't match");

  return EXIT_SUCCESS;
}
