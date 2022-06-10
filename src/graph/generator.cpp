#include "generator.h"

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cinttypes>
#include <iterator>
#include <memory>
#include <omp.h>
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

  IndexVec index = IndexVector::create(last_zero_idx + 1);
  NodeVec neighbors = NodeVector::create(offsets[last_zero_idx]);

  // No point in making this parallel since the vectors are
  // one-writer-multiple-readers
#pragma omp parallel for
  for (NodeT n = 0; n < last_zero_idx + 1; n++)
    index.get()[n] = offsets[n];

#pragma omp parallel for
  for (NodeT n = 0; n < g.num_nodes(); n++) {
    if (g.in_degree(n) != 0) {
      auto u = rename[n]; // New name
      for (const auto vw : g.in_neigh(n))
        neighbors.get()[offsets[u]++] = rename[vw.v];
      std::sort(neighbors.get() + index.get()[u],
                neighbors.get() + index.get()[u + 1]);
    }
  }

  return std::shared_ptr<Graph>(new Graph(index, neighbors, last_zero_idx));
}

FeatureVec generate_features(NodeT num_nodes, int num_features, FeatureT min,
                             FeatureT max, unsigned seed) {
  constexpr IndexT batch_size = 1024;
  std::uniform_real_distribution<FeatureT> dist(min, max);

  IndexT num_elems = num_nodes * num_features;
  FeatureVec features = FeatureVector::create(num_elems);

  IndexT num_batches = (num_elems + batch_size - 1) / batch_size;
#pragma omp parallel firstprivate(dist)
  {
    auto tid = omp_get_thread_num();
    std::mt19937 gen(seed + tid);

    for (auto b = tid; b < num_batches; b += omp_get_num_threads())
      for (IndexT i = b * batch_size;
           i < std::min((b + 1) * batch_size, num_elems); i++)
        features.get()[i] = dist(gen);
  }

  return features;
}

GraphPtr generate_graph_sparsity(NodeT num_nodes, double sparsity,
                                 unsigned seed) {

  auto g = GraphPtr(new Graph());
  g->num_idx_nodes = num_nodes;
  g->num_neighbors = num_nodes;

  const size_t w_num_nodes = num_nodes;
  const size_t num_edges = w_num_nodes * w_num_nodes * sparsity;
  g->index = IndexVector::create(num_nodes + 1);
  g->neighbors = NodeVector::create(num_edges);

  // Parallel structure to allocate next set of neighbors
  using EntryT = struct {
    NodeT next_node;
    IndexT next_ngh_start;
  };
  std::atomic<EntryT> next_entry;
  next_entry.store({0, 0});

// Randomly popluate neighbors
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto num_threads = omp_get_num_threads();

    std::mt19937_64 gen(seed + tid);
    std::uniform_real_distribution<double> sparse(0, 1);
    for (auto idx = tid; idx < num_nodes; idx += num_threads) {
      // Generate array
      std::vector<NodeT> neighs;
      for (NodeT ngh = 0; ngh < num_nodes; ngh++)
        if (sparse(gen) < sparsity)
          neighs.push_back(ngh);

      // Try to allocate a spot
      NodeT node;
      IndexT ngh_start;
      IndexT new_ngh_start;
      {
        EntryT entry, new_entry;
        do {
          entry = next_entry.load(std::memory_order::memory_order_relaxed);
          node = entry.next_node;
          ngh_start = entry.next_ngh_start;
          new_ngh_start = std::min(ngh_start + neighs.size(), num_edges);
          new_entry = {node + 1, new_ngh_start};
        } while (!std::atomic_compare_exchange_weak_explicit(
            &next_entry, &entry, new_entry,
            std::memory_order::memory_order_release,
            std::memory_order::memory_order_relaxed));
      }

      // Update index wih new spot
      g->index.get()[node] = ngh_start;

      // Populate neighbors
      for (IndexT i = ngh_start; i < new_ngh_start; i++)
        g->neighbors.get()[i] = neighs[i - ngh_start];
    }
  }

  // Set the final value of the index array
  g->index.get()[num_nodes] = g->index.get()[num_nodes - 1];
  assert(g->index.get()[num_nodes] <= num_edges);

  return g;
}
