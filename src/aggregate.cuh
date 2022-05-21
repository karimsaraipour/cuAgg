#ifndef SRC__AGGREGATE_CUH
#define SRC__AGGREGATE_CUH

#include "graph.h"

/**
 * Dummy GPU implementation of aggregate (proof-of-concept)
 * Note: It's missing the output feature array.
 */
__global__ void dummy_aggregate_kernel(const IndexT *const index,
                                       const NodeT *const neighbors,
                                       const FeatureT *const features,
                                       const NodeT num_nodes,
                                       const int num_features);

/**
 * Parallel CPU implementation of aggregate.
 * Used for validation of GPU kernels.
 */
void aggregate_cpu(const GraphPtr g, const FeatureVec &features,
                   FeatureVec &out_features, int num_features);

#endif // SRC__AGGREGATE_CUH
