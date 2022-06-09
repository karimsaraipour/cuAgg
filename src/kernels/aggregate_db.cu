#include "aggregate.cuh"

#include "../cuda.cuh"

#include <assert.h>

void aggregate_double_buffer_naive(const PartitionVec partitions,
                                   const NodeT num_tiles1D,
                                   const FeatureVec &in_features,
                                   FeatureT *const out_features,
                                   const IndexT num_features,
                                   const NodeT tile_size, const int db_size) {
  // Allocate GPU memory here
  IndexT **cu_index = new IndexT *[db_size];
  NodeT **cu_neighbors = new NodeT *[db_size];
  FeatureT **cu_in_features = new FeatureT *[db_size];
  FeatureT *cu_out_features;

  size_t size_index = (tile_size + 1) * sizeof(IndexT);
  size_t size_neighbors = tile_size * tile_size * sizeof(NodeT);
  size_t size_features = tile_size * num_features * sizeof(FeatureT);

  for (int i = 0; i < db_size; i++) {
    CUDA_ERRCHK(cudaMalloc((void **)&cu_index[i], size_index));
    CUDA_ERRCHK(cudaMalloc((void **)&cu_neighbors[i], size_neighbors));
    CUDA_ERRCHK(cudaMalloc((void **)&cu_in_features[i], size_features));
  }
  CUDA_ERRCHK(cudaMalloc((void **)&cu_out_features, size_features));

  // Helper function
  auto load_buffer_and_execute = [&](int b, const NodeT idx_tile,
                                     const NodeT ngh_tile) -> void {
    auto part = partitions[idx_tile * num_tiles1D + ngh_tile];

    // Load input features
    auto size_part_infeats =
        part.subgraph->num_neighbors * num_features * sizeof(FeatureT);
    CUDA_ERRCHK(cudaMemcpyAsync(
        cu_in_features[b],
        &in_features.data()[ngh_tile * tile_size * num_features],
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
    aggregate_dyn<<<tile_size, 32>>>(cu_index[b], cu_neighbors[b],
                                     cu_in_features[b], cu_out_features,
                                     subg->num_idx_nodes, num_features);
  };

  // Execute kernel
  int b = 0;
  for (NodeT idx_tile = 0; idx_tile < num_tiles1D; idx_tile++) {
    // Load output features
    auto part = partitions[idx_tile * num_tiles1D];
    auto size_part_outfeats =
        part.subgraph->num_idx_nodes * num_features * sizeof(FeatureT);
    CUDA_ERRCHK(cudaMemcpyAsync(
        cu_out_features, &out_features[idx_tile * tile_size * num_features],
        size_part_outfeats, cudaMemcpyHostToDevice));

    // Execute each input tile
    for (NodeT ngh_tile = 0; ngh_tile < num_tiles1D; ngh_tile++) {
      load_buffer_and_execute(b, idx_tile, ngh_tile);
      b = (b + 1) % db_size; // Switch buffer
    }

    // Unload output features
    CUDA_ERRCHK(cudaMemcpyAsync(
        &out_features[idx_tile * tile_size * num_features], cu_out_features,
        size_part_outfeats, cudaMemcpyDeviceToHost));
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
