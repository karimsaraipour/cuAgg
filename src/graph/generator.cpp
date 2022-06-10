#include "generator.h"

#include <algorithm>
#include <assert.h>
#include <cinttypes>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../../deps/gapbs/src/builder.h"
#include "../../deps/gapbs/src/graph.h"

GraphPtr generate_krongraph(int scale, int degree) {
  using WeightT = float;
  using WNodeT = NodeWeight<NodeT, WeightT>;
  using Builder = BuilderBase<NodeT, WNodeT, WeightT>;

  // Emulate command line arguments.
  std::vector<std::string> cpp_argv = {"execbin", "-g", std::to_string(scale),
                                       "-k", std::to_string(degree)};
  char **argv = new char *[cpp_argv.size()];
  for (int i = 0; i < cpp_argv.size(); i++)
    argv[i] = const_cast<char *>(cpp_argv[i].c_str());
  CLBase cli(cpp_argv.size(), argv);
  assert(cli.ParseArgs());
  delete[] argv;

  // Generate graph
  Builder b(cli);
  auto g = b.MakeGraph();

  // Sort by descending degree
  using DegreeNodePair = std::pair<NodeT, NodeT>;
  pvector<DegreeNodePair> degree_nid_pairs(g.num_nodes());
#pragma omp parallel for
  for (NodeT n = 0; n < g.num_nodes(); n++)
    degree_nid_pairs[n] = std::make_pair(g.in_degree(n), n);
  std::sort(degree_nid_pairs.begin(), degree_nid_pairs.end(),
            std::greater<DegreeNodePair>());

  // Rename nodes
  pvector<NodeT> degrees(g.num_nodes());
  pvector<NodeT> rename(g.num_nodes());
#pragma omp parallel for
  for (NodeT n = 0; n < g.num_nodes(); n++) {
    degrees[n] = degree_nid_pairs[n].first;
    rename[degree_nid_pairs[n].second] = n;
  }

  // Figure out last non-zero node
  assert(degrees[0] != 0 && "At least two nodes should have edges");
  NodeT last_zero_idx = g.num_nodes();
  while (last_zero_idx != 0 && degrees[last_zero_idx - 1] == 0)
    last_zero_idx--;

  // Rebuild graph & squash zeros
  auto offsets = Builder::ParallelPrefixSum(degrees);

  IndexVec index(last_zero_idx + 1);
  NodeVec neighbors(offsets[last_zero_idx]);

  // No point in making this parallel since the vectors are
  // one-writer-multiple-readers
  for (NodeT n = 0; n < last_zero_idx + 1; n++)
    index[n] = offsets[n];

  for (NodeT n = 0; n < g.num_nodes(); n++) {
    if (g.in_degree(n) != 0) {
      auto u = rename[n]; // New name
      for (const auto vw : g.in_neigh(n))
        neighbors[offsets[u]++] = rename[vw.v];
      std::sort(neighbors.begin() + index[u], neighbors.begin() + index[u + 1]);
    }
  }

  return std::shared_ptr<Graph>(new Graph(index, neighbors, last_zero_idx));
}

FeatureVec generate_features(NodeT num_nodes, int num_features, FeatureT min,
                             FeatureT max, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<FeatureT> dist(min, max);

  IndexT num_elems = num_nodes * num_features;
  FeatureVec features(num_elems);
  for (IndexT i = 0; i < num_elems; i++)
    features[i] = dist(gen);

  return features;
}

GraphPtr generate_graph_sparsity(NodeT num_nodes, double sparsity,
                                 unsigned seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(0, 1);

  auto g = GraphPtr(new Graph());
  g->num_idx_nodes = num_nodes;
  g->num_neighbors = num_nodes;
  g->index.push_back(0);

  // Generate based on probability
  // (theoretically guaranteed to have num_nodes^2 * sparsity edges)
  for (NodeT v = 0; v < num_nodes; v++) {
    // Randomly populate neighbors
    for (NodeT u = 0; u < num_nodes; u++) {
      double prob = dist(gen);
      if (prob < sparsity)
        g->neighbors.push_back(u);
    }

    // Update index
    g->index.push_back(g->neighbors.size());
  }

  return g;
}
