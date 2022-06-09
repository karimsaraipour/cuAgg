#ifndef SRC_KERNELS__AGGREGATE_CUH
#define SRC_KERNELS__AGGREGATE_CUH

#include "../graph/graph.h"
#include "../graph/partition.h"

/**
 * Naive aggregate kernel.
 * Each thread is assigned a node (offset by num thread.X) and a feature
 * (offset by num thread.Y).
 */
__global__ void
aggregate_naive(const IndexT *const index, const NodeT *const neighbors,
                const FeatureT *const in_features, FeatureT *const out_features,
                const NodeT num_nodes, const IndexT num_features);

/**
 * Dynamic aggergate kernel.
 * Each thread block is assigned to a node (offset by num thread blocks) and
 * given N warps.
 * Each of the N warps will work on one neighbor (offset by N).
 *
 * Assumption:
 *  Initialize kernel with threads % warpSize == 0
 *
 * Usage:
 *  aggregate_dyn<<<NUMBER OF THREAD BLOCKS, WARPS PER NODE * warpSize>>>(...)
 */
__global__ void aggregate_dyn(const IndexT *const index,
                              const NodeT *const neighbors,
                              const FeatureT *const in_features,
                              FeatureT *const out_features,
                              const NodeT num_nodes, const IndexT num_features);

/**
 * Parallel CPU implementation of aggregate.
 * Used for validation of GPU kernels.
 */
void aggregate_cpu(const GraphPtr g, const FeatureVec &in_features,
                   FeatureVec &out_features, IndexT num_features);

/**
 * Double buffer naive partitioning
 */
void aggregate_double_buffer_naive(const PartitionVec partitions,
                                   const NodeT num_tiles1D,
                                   const FeatureVec &in_features,
                                   FeatureT *const out_features,
                                   const IndexT num_features,
                                   const NodeT tile_size);

#endif // SRC__AGGREGATE_CUH
