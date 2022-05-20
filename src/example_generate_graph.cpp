#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>

#include "generator.h"
#include "graph.h"

int main(int argc, char *argv[]) {
  constexpr int SCALE = 4;  // 2^{scale} number of nodes
  constexpr int DEGREE = 3; // average degree
  constexpr int NUM_FEATURES = 16;
  constexpr NodeT PRINT_NODE_LIMIT = 5;
  constexpr IndexT PRINT_EDGE_LIMIT = 10;

  // Generate graph
  auto g = generate_krongraph(SCALE, DEGREE);

  // Generate feature vectors
  auto features = generate_features(g->num_nodes, NUM_FEATURES);

  std::cout << "Features" << std::endl;
  for (NodeT n = 0; n < std::min(g->num_nodes, PRINT_NODE_LIMIT); n++) {
    std::cout << "Node " << n << ": ";
    for (IndexT f = 0; f < NUM_FEATURES; f++)
      std::cout << features[n * NUM_FEATURES + f] << " ";
    std::cout << std::endl;
  }

  // (De)serialization
  std::ofstream ofs("my.g", std::ofstream::out);
  ofs << *g;
  ofs.close();

  GraphPtr g_read = std::unique_ptr<Graph>(new Graph);
  std::ifstream ifs("my.g", std::ifstream::in);
  ifs >> *g_read;
  ifs.close();

  // Print out the input graph
  {
    IndexT edges = 0;
    std::cout << "Original graph" << std::endl;
    for (NodeT u = 0; u < g->num_nodes; u++) {
      for (IndexT i = g->index[u]; i < g->index[u + 1]; i++) {
        std::cout << u << " -> " << g->neighbors[i] << std::endl;

        edges++;
        if (edges == PRINT_EDGE_LIMIT)
          break;
      }
      if (edges == PRINT_EDGE_LIMIT)
        break;
    }
  }

  {
    IndexT edges = 0;
    std::cout << "Read graph" << std::endl;
    for (NodeT u = 0; u < g_read->num_nodes; u++) {
      for (IndexT i = g_read->index[u]; i < g_read->index[u + 1]; i++) {
        std::cout << u << " -> " << g_read->neighbors[i] << std::endl;

        edges++;
        if (edges == PRINT_EDGE_LIMIT)
          break;
      }
      if (edges == PRINT_EDGE_LIMIT)
        break;
    }
  }

  return EXIT_SUCCESS;
}
