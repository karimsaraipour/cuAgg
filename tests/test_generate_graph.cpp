#include <assert.h>
#include <cstdlib>
#include <fstream>

#include "../src/graph/generator.h"
#include "../src/graph/graph.h"

int main(int argc, char *argv[]) {
  constexpr int TEST_SCALE = 14;
  constexpr int TEST_DEGREE = 10;
  constexpr int NUM_FEATURES = 16;

  // Generate graph
  auto g = generate_krongraph(TEST_SCALE, TEST_DEGREE);

  // (De)serialization
  std::ofstream ofs("my.g", std::ofstream::out);
  ofs << *g;
  ofs.close();

  GraphPtr g_read = GraphPtr(new Graph());
  std::ifstream ifs("my.g", std::ifstream::in);
  ifs >> *g_read;
  ifs.close();

  // Make sure metadata is the same
  assert(g->num_idx_nodes == g_read->num_idx_nodes);
  assert(g->num_neighbors == g_read->num_neighbors);

  // Make sure index arrays match
  for (NodeT idx = 0; idx < g->num_idx_nodes + 1; idx++)
    assert(g->index[idx] == g_read->index[idx]);

  // Make sure neighbor arrays match
  for (IndexT idx = 0; idx < g->index[g->num_idx_nodes]; idx++)
    assert(g->neighbors[idx] == g_read->neighbors[idx]);

  return EXIT_SUCCESS;
}
