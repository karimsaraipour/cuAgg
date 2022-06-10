#include "aggregate.cuh"

#include <assert.h>
#include <math.h>

#include "../cuda.cuh"
#include "../graph/partition.h"

void aggregate_double_buffer_naive(
    const PartitionVec partitions, const NodeT num_idx_tiles,
    const FeatureVec &in_features, FeatureT *const out_features,
    const IndexT num_features, const NodeT tile_size, AggregateFunc kernel,
    const int db_size, const size_t neighbors_size) {
  for (const auto &part : partitions) {
    assert(part.idx_map.type == NodeMapping::MappingT::affine);
    assert(part.idx_map.type == NodeMapping::MappingT::affine);
  }
  NodeT num_ngh_tiles = partitions.size() / num_idx_tiles;

  // Allocate GPU memory here
  IndexT **cu_index = new IndexT *[db_size];
  NodeT **cu_neighbors = new NodeT *[db_size];
  FeatureT **cu_in_features = new FeatureT *[db_size];
  FeatureT *cu_out_features;

  size_t size_index = (tile_size + 1) * sizeof(IndexT);
  size_t size_neighbors =
      ((neighbors_size == 0) ? tile_size * tile_size : neighbors_size) *
      sizeof(NodeT);
  size_t size_features = tile_size * num_features * sizeof(FeatureT);
  std::cout << size_index << std::endl
            << size_neighbors << std::endl
            << size_features << std::endl
            << (size_index + size_neighbors + size_features) * db_size +
                   size_features
            << std::endl;

  for (int i = 0; i < db_size; i++) {
    CUDA_ERRCHK(cudaMalloc((void **)&cu_index[i], size_index));
    CUDA_ERRCHK(cudaMalloc((void **)&cu_neighbors[i], size_neighbors));
    CUDA_ERRCHK(cudaMalloc((void **)&cu_in_features[i], size_features));
  }
  CUDA_ERRCHK(cudaMalloc((void **)&cu_out_features, size_features));

  // Helper function
  auto load_buffer_and_execute = [&](int b, const NodeT idx_tile,
                                     const NodeT ngh_tile) -> void {
    auto part = partitions[idx_tile * num_ngh_tiles + ngh_tile];

    // Load input features
    auto size_part_infeats =
        part.subgraph->num_neighbors * num_features * sizeof(FeatureT);
    CUDA_ERRCHK(
        cudaMemcpyAsync(cu_in_features[b],
                        &in_features.data()[part.ngh_map.base * num_features],
                        size_part_infeats, cudaMemcpyHostToDevice));

    auto subg = part.subgraph;
    auto size_part_idx = (subg->num_idx_nodes + 1) * sizeof(IndexT);
    auto size_part_nghs = subg->index[subg->num_idx_nodes] * sizeof(NodeT);

    // Load index
    CUDA_ERRCHK(cudaMemcpyAsync(cu_index[b], subg->index.data(), size_part_idx,
                                cudaMemcpyHostToDevice));

    // Load neighbors
    CUDA_ERRCHK(cudaMemcpyAsync(cu_neighbors[b], subg->neighbors.data(),
                                size_part_nghs, cudaMemcpyHostToDevice));

    // Execute kernel
    kernel(cu_index[b], cu_neighbors[b], cu_in_features[b], cu_out_features,
           subg->num_idx_nodes, num_features);
  };

  // Execute kernel
  int b = 0;
  for (NodeT idx_tile = 0; idx_tile < num_idx_tiles; idx_tile++) {
    // Load output features
    auto part = partitions[idx_tile * num_ngh_tiles];
    auto size_part_outfeats =
        part.subgraph->num_idx_nodes * num_features * sizeof(FeatureT);
    CUDA_ERRCHK(cudaMemcpyAsync(cu_out_features,
                                &out_features[part.idx_map.base * num_features],
                                size_part_outfeats, cudaMemcpyHostToDevice));

    // Execute each input tile
    for (NodeT ngh_tile = 0; ngh_tile < num_ngh_tiles; ngh_tile++) {
      load_buffer_and_execute(b, idx_tile, ngh_tile);
      b = (b + 1) % db_size; // Switch buffer
    }

    // Unload output features
    CUDA_ERRCHK(cudaMemcpyAsync(&out_features[part.idx_map.base * num_features],
                                cu_out_features, size_part_outfeats,
                                cudaMemcpyDeviceToHost));
  }

  // Free memory
  for (int i = 0; i < db_size; i++) {
    CUDA_ERRCHK(cudaFree(cu_index[i]));
    CUDA_ERRCHK(cudaFree(cu_neighbors[i]));
    CUDA_ERRCHK(cudaFree(cu_in_features[i]));
  }
  CUDA_ERRCHK(cudaFree(cu_out_features));

  delete[] cu_index;
  delete[] cu_neighbors;
  delete[] cu_in_features;
}

NodeT get_square_tile_size(const IndexT num_features, const int db_size,
                           const float sparsity) {
  assert(sparsity > 0.000001); // != 0 hack

  cudaDeviceProp prop;
  CUDA_ERRCHK(cudaGetDeviceProperties(&prop, 0));

  size_t total_mem = prop.totalGlobalMem * 0.9f;
  std::cout << total_mem << std::endl;

  // Assumption that neighbors array is not sparse will hurt the general
  // case!
  auto memory_used = [num_features, db_size,
                      sparsity](const size_t num_nodes) -> size_t {
    size_t size_feature = num_nodes * num_features * sizeof(FeatureT);
    size_t size_index = (num_nodes + 1) * sizeof(IndexT);
    size_t size_neighbors = (num_nodes * num_nodes) * sizeof(NodeT) * sparsity;
    return size_feature +
           (size_feature + size_index + size_neighbors) * db_size;
  };

  // Do binary search to find best point
  NodeT start = 0;
  NodeT end = std::sqrt(total_mem / sizeof(NodeT) / db_size /
                        sparsity); // conservative estimate

  auto mem_use = memory_used(end);
  assert(total_mem <= memory_used(end));

  while (start < end) {
    NodeT mid = (start + end) / 2;
    if (memory_used(mid) > total_mem) {
      end = mid;
    } else {
      start = mid + 1;
    }
  }

  // Could still be possible that the final tile size is one too large
  if (memory_used(start) > total_mem) {
    assert(memory_used(start - 1) <= total_mem);
    std::cout << memory_used(start - 1) << std::endl;
    return start - 1;
  }

  std::cout << memory_used(start - 1) << std::endl;
  return start;
}
