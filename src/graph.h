#ifndef SRC__GRAPH_H
#define SRC__GRAPH_H

#include <memory>
#include <vector>

struct Graph;
using NodeT = int;
using IndexT = int;
using NodeVecT = std::vector<NodeT>;
using IndexVecT = std::vector<NodeT>;
using GraphPtr = std::unique_ptr<Graph>;

/**
 * Basic CSR structure for graphs.
 *
 * Accessing neighbors of node u for graph g
 * for (IndexT i = g;index[u]; i < g.index[u + 1]; i++) {
 *   NodeT v = g.neighbors[i]; // Edge from u to v
 * }
 */
struct Graph {
  IndexVecT index;    // size = N + 1
  NodeVecT neighbors; // size = M
  NodeT num_nodes;    // number of nodes
};

#endif // SRC__GRAPH_H
