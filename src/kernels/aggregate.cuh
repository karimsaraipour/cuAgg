#ifndef SRC_KERNELS__AGGREGATE_CUH
#define SRC_KERNELS__AGGREGATE_CUH

#include <functional>

#include "../graph/graph.h"
#include "../graph/partition.h"

using AggregateFunc = std::function<void(
    const IndexT *const, const NodeT *const, const FeatureT *const,
    FeatureT *const, const NodeT, const IndexT)>;

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
 * num_idx_tiles indicates how many index tiles are currently in the partition
 * Use db_size=1 for no double buffering
 */
void aggregate_double_buffer_naive(
    const PartitionVec partitions, const NodeT num_idx_tiles,
    const FeatureVec &in_features, FeatureT *const out_features,
    const IndexT num_features, const NodeT tile_size, AggregateFunc kernel,
    const int db_size = 2, const size_t neighbors_size = 0);

NodeT get_square_tile_size(const IndexT num_features, const int db_size = 2,
                           const float sparsity = 0.2);

#endif // SRC__AGGREGATE_CUH
