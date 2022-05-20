#ifndef SRC__AGGREGATE_CUH
#define SRC__AGGREGATE_CUH

#include "graph.h"

__global__ void dummy_aggregate_kernel(const IndexT * const index,
                const NodeT * const neighbors, const FeatureT * const features,
                const NodeT num_nodes, const int num_features);

#endif // SRC__AGGREGATE_CUH
