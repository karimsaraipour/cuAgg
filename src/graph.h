#ifndef SRC__GRAPH_H
#define SRC__GRAPH_H

#include <iostream>
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

  Graph() : Graph(IndexVecT(), NodeVecT(), 0) {}
  Graph(IndexVecT index_, NodeVecT neighbors_, NodeT num_nodes_)
      : index(index_), neighbors(neighbors_), num_nodes(num_nodes_) {}
};

/**
 * Serialization
 * Usage: file << g;
 */
std::ostream &operator<<(std::ostream &os, Graph &g);

/**
 * Deserialization
 * Usage: file >> g;
 */
std::istream &operator>>(std::istream &is, Graph &g);

#endif // SRC__GRAPH_H
