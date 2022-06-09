#include "aggregate.cuh"

#include <stdio.h>

__global__ void
aggregate_naive(const IndexT *const index, const NodeT *const neighbors,
                const FeatureT *const in_features, FeatureT *const out_features,
                const NodeT num_nodes, const IndexT num_features) {
  NodeT start_v = blockIdx.x * blockDim.x + threadIdx.x;
  NodeT offset_v = blockDim.x * gridDim.x;
  IndexT start_f = blockIdx.y * blockDim.y + threadIdx.y;
  IndexT offset_f = blockDim.y * gridDim.y;

  for (NodeT v = start_v; v < num_nodes; v += offset_v) {
    // Write out output feature (itself)
    for (IndexT f = start_f; f < num_features; f += offset_f)
      out_features[v * num_features + f] = 0;

    // Aggregate neighbors
    for (IndexT i = index[v]; i < index[v + 1]; i++) {
      NodeT u = neighbors[i]; // Edges go from u->v
      for (IndexT f = start_f; f < num_features; f += offset_f)
        out_features[v * num_features + f] += in_features[u * num_features + f];
    }
  }
}

// CSR -> row_idx, col_idx
// index = row_idx, neighbors = col_idx
// index[num_nodes + 1]
__global__ void
aggregate_dyn(const IndexT *const index, const NodeT *const neighbors,
              const FeatureT *const in_features, FeatureT *const out_features,
              const NodeT num_nodes, const IndexT num_features) {
  // Compute indices
  NodeT start_v = blockIdx.x;
  NodeT offset_v = gridDim.x;
  int warp_id = threadIdx.x / warpSize; // Which warp in the thread block
  int warp_lane =
      threadIdx.x % warpSize; // Which thread in the warp [0, warpSize)
  int offset_i = blockDim.x / warpSize; // How many warps in a thread block
  int offset_f = warpSize;

  // iterate over destination nodes
  for (NodeT v = start_v; v < num_nodes; v += offset_v) {
    // aggregate
    // each warp in thread block handles a neighbor
    // the for loop is strided
    // threads in a warp handles features in embedding
    for (IndexT i = index[v] + warp_id; i < index[v + 1]; i += offset_i) {
      NodeT u = neighbors[i]; // Edges go from u->v
      for (IndexT f = warp_lane; f < num_features; f += offset_f)
        atomicAdd(&out_features[v * num_features + f],
                  in_features[u * num_features + f]);
    }
  }
}

// CSR -> row_idx, col_idx
// index = row_idx, neighbors = col_idx
// index[num_nodes + 1]
__global__ void aggregate_dyn_sm(const IndexT *const index,
                                 const NodeT *const neighbors,
                                 const FeatureT *const in_features,
                                 FeatureT *const out_features,
                                 const NodeT num_nodes,
                                 const IndexT num_features) {
  // Compute indices
  NodeT start_v = blockIdx.x;
  NodeT offset_v = gridDim.x;
  int warp_id = threadIdx.x / warpSize; // Which warp in the thread block
  int warp_lane =
      threadIdx.x % warpSize; // Which thread in the warp [0, warpSize)
  int offset_i = blockDim.x / warpSize; // How many warps in a thread block
  int offset_f = warpSize;

  // // iterate over destination nodes
  // for (NodeT v = start_v; v < num_nodes; v += offset_v) {

  //   // copy of tile of adjancency matrix
  //   if (warp_id == 0) // First warp
  //     for (IndexT f = warp_lane; f < num_features; f += offset_f)
  //       out_features[v * num_features + f] = in_features[v * num_features +
  //       f];
  //   __syncthreads();

  //   // aggregate
  //   // each warp in thread block handles a neighbor
  //   // the for loop is strided
  //   // threads in a warp handles features in embedding
  //   for (IndexT i = index[v] + warp_id; i < index[v + 1]; i += offset_i) {
  //     NodeT u = neighbors[i]; // Edges go from u->v
  //     for (IndexT f = warp_lane; f < num_features; f += offset_f)
  //       atomicAdd(&out_features[v * num_features + f],
  //                 in_features[u * num_features + f]);
  //   }
  // }

  // aggregate_dyn<<<num_blocks, num_threads, shared_mem_size>>>(...<whatever>);
  extern __shared__ FeatureT
      shared_out_features[]; // dynamically allocated at run time
  for (NodeT v = start_v; v < num_nodes; v += offset_v) { // out_neighbors
    for (IndexT f = warp_lane; f < num_features; f += offset_f) { // features
      // load a tile of the feature vectors
      shared_out_features[f] = out_features[v * num_features + f];
      __syncthreads();
      // aggregate using an atomic add
      for (IndexT i = index[v] + warp_id; i < index[v + 1]; i += offset_i)
        atomicAdd(&shared_out_features[f],
                  in_features[neighbors[i] * num_features + f]);

      out_features[v * num_features + f] = shared_out_features[f];
      __syncthreads();
    }
  }
}
