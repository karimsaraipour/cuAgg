#ifndef SRC__AGGREGATE_CUH
#define SRC__AGGREGATE_CUH

#include "graph.h"

/**
 * Naive aggregate kernel
 * Each thread is assigned a vertex (offset by num thread.X) and a feature
 * (offset by num thread.Y).
 */
__global__ void
aggregate_naive(const IndexT *const index, const NodeT *const neighbors,
                const FeatureT *const in_features, FeatureT *const out_features,
                const NodeT num_nodes, const IndexT num_features);

/**
 * Parallel CPU implementation of aggregate.
 * Used for validation of GPU kernels.
 */
void aggregate_cpu(const GraphPtr g, const FeatureVec &in_features,
                   FeatureVec &out_features, IndexT num_features);

#endif // SRC__AGGREGATE_CUH
