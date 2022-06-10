#include "graph.h"
#include <memory>

std::ostream &operator<<(std::ostream &os, Graph &g) {
  // Write metadata
  auto num_idx_nodes = g.num_idx_nodes;
  auto num_neighbors = g.num_neighbors;
  auto num_edges = g.index.get()[g.num_idx_nodes];
  os.write(reinterpret_cast<char *>(&num_idx_nodes), sizeof(NodeT));
  os.write(reinterpret_cast<char *>(&num_neighbors), sizeof(NodeT));
  os.write(reinterpret_cast<char *>(&num_edges), sizeof(IndexT));

  // Write graph
  os.write(reinterpret_cast<char *>(g.index.get()),
           (num_idx_nodes + 1) * sizeof(IndexT));
  os.write(reinterpret_cast<char *>(g.neighbors.get()),
           num_edges * sizeof(NodeT));

  return os;
}

std::istream &operator>>(std::istream &is, Graph &g) {
  // Read metadata
  NodeT num_idx_nodes;
  NodeT num_neighbors;
  IndexT num_edges;
  is.read(reinterpret_cast<char *>(&num_idx_nodes), sizeof(NodeT));
  is.read(reinterpret_cast<char *>(&num_neighbors), sizeof(NodeT));
  is.read(reinterpret_cast<char *>(&num_edges), sizeof(IndexT));

  // Resize graph structures
  g.num_idx_nodes = num_idx_nodes;
  g.num_neighbors = num_neighbors;
  g.index = IndexVector::create(num_idx_nodes + 1);
  g.neighbors = NodeVector::create(num_edges);

  // Read graph
  is.read(reinterpret_cast<char *>(g.index.get()),
          (num_idx_nodes + 1) * sizeof(IndexT));
  is.read(reinterpret_cast<char *>(g.neighbors.get()),
          num_edges * sizeof(NodeT));

  return is;
}
