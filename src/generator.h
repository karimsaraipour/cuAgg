#ifndef SRC__GENERATOR_H
#define SRC__GENERATOR_H

#include "graph.h"

/**
 * Generates a Kronecker graph with a given scale and average degree.
 *
 * Number of nodes ~= 2^{scale} (will actually be around half since degree-0
 * nodes get pruned)
 * Number of edges = 2^{scale} * degree
 */
GraphPtr generate_krongraph(int scale, int degree);

/**
 * Generate feature vectors.
 * Indexing convention: features[node id * number of features + feature id]
 */
FeatureVecT generate_features(NodeT num_nodes, int num_features,
                              FeatureT min = 0, FeatureT max = 1024,
                              unsigned seed = 64);

#endif // SRC__GENERATOR_H
