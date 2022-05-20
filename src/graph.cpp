#include "graph.h"

std::ostream &operator<<(std::ostream &os, Graph &g) {
  // Write metadata
  auto num_nodes = g.num_nodes;
  auto num_edges = g.index[g.num_nodes];
  os.write(reinterpret_cast<char *>(&num_nodes), sizeof(NodeT));
  os.write(reinterpret_cast<char *>(&num_edges), sizeof(IndexT));

  // Write graph
  os.write(reinterpret_cast<char *>(g.index.data()),
           (num_nodes + 1) * sizeof(IndexT));
  os.write(reinterpret_cast<char *>(g.neighbors.data()),
           num_edges * sizeof(NodeT));

  return os;
}

std::istream &operator>>(std::istream &is, Graph &g) {
  // Read metadata
  NodeT num_nodes;
  IndexT num_edges;
  is.read(reinterpret_cast<char *>(&num_nodes), sizeof(NodeT));
  is.read(reinterpret_cast<char *>(&num_edges), sizeof(IndexT));

  std::cout << num_nodes << " " << num_edges << std::endl;

  // Resize graph structures
  g.num_nodes = num_nodes;
  g.index.resize(num_nodes + 1);
  g.neighbors.resize(num_edges);

  // Read graph
  is.read(reinterpret_cast<char *>(g.index.data()),
          (num_nodes + 1) * sizeof(IndexT));
  is.read(reinterpret_cast<char *>(g.neighbors.data()),
          num_edges * sizeof(NodeT));

  return is;
}
