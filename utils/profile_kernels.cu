#include <cstdlib>
#include <fstream>
#include <sstream>

#include "../src/graph/generator.h"
#include "../src/graph/graph.h"

int main(int argc, char *argv[]) {
  double sparsity = atof(argv[1]);
  auto g = generate_graph_sparsity(512, sparsity);

  std::stringstream ss;
  ss << "graph_sparsity" << argv[1] << ".g";

  std::ofstream ofs(ss.str(), std::ofstream::out);
  ofs << *g;
  ofs.close();

  return EXIT_SUCCESS;
}
