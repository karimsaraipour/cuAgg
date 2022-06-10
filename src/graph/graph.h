#ifndef SRC_GRAPH__GRAPH_H
#define SRC_GRAPH__GRAPH_H

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

struct Graph;
using NodeT = int;
using IndexT = int;
using FeatureT = float;
using NodeVec = std::shared_ptr<NodeT>;
using IndexVec = std::shared_ptr<IndexT>;
using FeatureVec = std::shared_ptr<FeatureT>;
using GraphPtr = std::shared_ptr<Graph>;

template <typename T> struct Vector {
  using VecT = std::shared_ptr<T>;
  static VecT create(const size_t size) {
    return VecT(new T[size], std::default_delete<T[]>());
  }
};

using NodeVector = Vector<NodeT>;
using IndexVector = Vector<IndexT>;
using FeatureVector = Vector<FeatureT>;

/**
 * Basic CSR structure for graphs.
 *
 * Accessing the in-neighbors of node v for graph g
 * for (IndexT i = g;index[v]; i < g.index[v + 1]; i++) {
 *   NodeT u = g.neighbors[i]; // Edge from u to v
 * }
 *
 * Ultimately, we need to distinguish the size of the index array and
 * the number of neighbors since the CSR format represents a matrix. It is
 * possible for a matrix to not be square.
 */
struct Graph {
  IndexVec index;
  NodeVec neighbors;

  // This is essentially matrix rows & matrix columns respectively
  NodeT num_idx_nodes; // number of nodes indexd (i.e., index size)
  NodeT num_neighbors; // number of neighbors (i.e., max(neighbors))

  enum class DirectionT { push, pull } direction;

  Graph() : Graph(nullptr, nullptr, 0, DirectionT::pull) {}
  Graph(IndexVec index_, NodeVec neighbors_, NodeT num_nodes_,
        DirectionT direction_ = DirectionT::pull)
      : index(index_), neighbors(neighbors_), num_idx_nodes(num_nodes_),
        num_neighbors(num_nodes_), direction(direction_) {}
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

#endif // SRC_GRAPH__GRAPH_H
