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

#endif // SRC__GENERATOR_H
