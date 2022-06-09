#ifndef SRC_KERNELS__AGGREGATE_TEMPLATED_CU
#define SRC_KERNELS__AGGREGATE_TEMPLATED_CU

#include "../graph/graph.h"

/**
 * Dynamic aggergate kernel.
 * Each thread block is assigned to a node (offset by num thread blocks) and
 * given N warps.
 * Each of the N warps will work on one neighbor (offset by N).
 *
 * Warps will work together, saving the partial to shared memory before writing
 * it to global memory
 *
 * Assumption:
 *  Initialize kernel with threads % warpSize == 0
 *
 * Usage:
 *  aggregate_dyn<<<NUMBER OF THREAD BLOCKS, WARPS PER NODE * warpSize>>>(...)
 */
template <size_t feature_tile_size>
__global__ void
aggregate_dyn_sm(const IndexT *const index, const NodeT *const neighbors,
                 const FeatureT *const in_features,
                 FeatureT *const out_features, const NodeT num_nodes,
                 const IndexT num_features) {
  // Feature map tile
  __shared__ FeatureT sm_out_features[feature_tile_size];

  // Compute indices
  NodeT start_v = blockIdx.x;
  NodeT offset_v = gridDim.x;
  int warp_id = threadIdx.x / warpSize; // Which warp in the thread block
  int warp_lane =
      threadIdx.x % warpSize; // Which thread in the warp [0, warpSize)
  IndexT offset_i = blockDim.x / warpSize; // How many warps in a thread block
  IndexT offset_f = warpSize;
  IndexT feature_tiles =
      (num_features + feature_tile_size - 1) / feature_tile_size;

  // Aggregate neighbors
  for (NodeT v = start_v; v < num_nodes; v += offset_v) { // out_neighbors
    for (IndexT ft = 0; ft < feature_tiles; ft++) {       // feature tile
      auto ft_size =
          min(num_features - ft * feature_tile_size, feature_tile_size);

      // Reset feature tile
      for (IndexT f = threadIdx.x; f < ft_size; f += blockDim.x)
        sm_out_features[f] = 0;
      __syncthreads();

      // Aggregate neighbors
      // Each warp gets a neighbor
      for (IndexT i = index[v] + warp_id; i < index[v + 1]; i += offset_i) {
        NodeT u = neighbors[i];
        for (IndexT f = warp_lane; f < ft_size; f += offset_f)
          atomicAdd(&sm_out_features[f],
                    in_features[u * num_features + ft * feature_tile_size + f]);
      }
      __syncthreads();

      // Write to global memory
      for (IndexT f = threadIdx.x; f < ft_size; f += blockDim.x)
        atomicAdd(&out_features[v * num_features + ft * feature_tile_size + f],
                  sm_out_features[f]);
      // Note: No need to sync here since the each thread will only overwrite
      // the region they pushed to global memory
    }
  }
}

/**
 * Dynamic aggergate kernel.
 * Each thread block is assigned to a node (offset by num thread blocks) and
 * given N warps.
 * Each of the N warps will work on one neighbor (offset by N).
 *
 * Warps will work independently, each writing their partial result to memory
 *
 * Assumption:
 *  Initialize kernel with threads % warpSize == 0
 *
 * Usage:
 *  aggregate_dyn<<<NUMBER OF THREAD BLOCKS, WARPS PER NODE * warpSize>>>(...)
 */
template <size_t feature_tile_size>
__global__ void
aggregate_dyn_rf(const IndexT *const index, const NodeT *const neighbors,
                 const FeatureT *const in_features,
                 FeatureT *const out_features, const NodeT num_nodes,
                 const IndexT num_features) {
  // Feature map tile
  FeatureT rf_out_features[feature_tile_size];

  // Compute indices
  NodeT start_v = blockIdx.x;
  NodeT offset_v = gridDim.x;
  int warp_id = threadIdx.x / warpSize; // Which warp in the thread block
  int warp_lane =
      threadIdx.x % warpSize; // Which thread in the warp [0, warpSize)
  IndexT offset_i = blockDim.x / warpSize; // How many warps in a thread block
  IndexT offset_f = warpSize;
  IndexT feature_tiles =
      (num_features + feature_tile_size - 1) / feature_tile_size;

  // Aggregate neighbors
  for (NodeT v = start_v; v < num_nodes; v += offset_v) { // out_neighbors
    for (IndexT ft = 0; ft < feature_tiles; ft++) {       // feature tile
      auto ft_size =
          min(num_features - ft * feature_tile_size, feature_tile_size);

      // Reset feature tile
      for (IndexT f = warp_lane; f < ft_size; f += warpSize)
        rf_out_features[f] = 0;

      // Aggregate neighbors
      // Each warp gets a neighbor
      for (IndexT i = index[v] + warp_id; i < index[v + 1]; i += offset_i) {
        NodeT u = neighbors[i];
        for (IndexT f = warp_lane; f < ft_size; f += offset_f)
          rf_out_features[f] +=
              in_features[u * num_features + ft * feature_tile_size + f];
      }

      // Is this necesary to coalesce atomic update?
      __syncthreads();

      // Write to global memory
      for (IndexT f = warp_lane; f < ft_size; f += warpSize)
        atomicAdd(&out_features[v * num_features + ft * feature_tile_size + f],
                  rf_out_features[f]);
      // Note: No need to sync here since the each thread will only overwrite
      // the region they pushed to global memory
    }
  }
}

#endif // SRC_KERNELS__AGGREGATE_TEMPLATED_CU
