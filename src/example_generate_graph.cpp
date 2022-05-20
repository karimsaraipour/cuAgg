#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>

#include "generator.h"
#include "graph.h"

int main(int argc, char *argv[]) {
  int scale = 4;  // 2^{scale} number of nodes
  int degree = 3; // average degree
  auto g = generate_krongraph(scale, degree);

  std::ofstream ofs("my.g", std::ofstream::out);
  ofs << *g;
  ofs.close();

  GraphPtr g_read = std::unique_ptr<Graph>(new Graph);
  std::ifstream ifs("my.g", std::ifstream::in);
  ifs >> *g_read;
  ifs.close();

  // Print out the input graph
  constexpr IndexT PRINT_EDGE_LIMIT = 10;

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
