#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "../src/graph/generator.h"
#include "../src/graph/graph.h"

struct GraphProp {
  int scale;
  int degree;
  std::string ofname;
};

GraphProp parse_args(int argc, char *argv[]) {
  GraphProp prop = {-1, -1, ""};

// Parse command line args
#define USAGE_MSG                                                              \
  "Usage: " << argv[0] << " -g <scale> -k <degree> [-f out_file_name.g]"

  // If no arguments
  if (argc == 1) {
    std::cerr << USAGE_MSG << std::endl;
    exit(EXIT_FAILURE);
  }

  char opt;
  while ((opt = getopt(argc, argv, "g:k:f:h")) != -1) {
    switch (opt) {
    case 'g':
      prop.scale = atoi(optarg);
      break;
    case 'k':
      prop.degree = atoi(optarg);
      break;
    case 'f': {
      std::string ofname(optarg);

      // Verify extension is correct
      if (ofname.substr(ofname.size() - 2, 2) != ".g") {
        std::cerr << "Bad file extension" << std::endl;
        std::cerr << USAGE_MSG << std::endl;
        exit(EXIT_FAILURE);
      }

      prop.ofname = ofname;
      break;
    }
    case 'h':
      std::cout << USAGE_MSG << std::endl;
      exit(EXIT_SUCCESS);
    case '?':
      std::cerr << "Unrecognized option -" << opt << std::endl;
      std::cerr << USAGE_MSG << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // Verify scale & degree have been defined
  if (prop.scale == -1 || prop.degree == -1) {
    std::cerr << "Must define both scale & degree of graph" << std::endl;
    std::cerr << USAGE_MSG << std::endl;
    exit(EXIT_FAILURE);
  }

  // Generate default file name if needed
  if (prop.ofname.empty()) {
    std::stringstream ss;
    ss << "kron_scale" << prop.scale << "_degree" << prop.degree << ".g";
    prop.ofname = ss.str();
  }

#undef USAGE_MSG

  // Reset optarg since GAPBS also uses it to parse arguments
  optind = 1;

  return prop;
}

int main(int argc, char *argv[]) {
  auto prop = parse_args(argc, argv);

  // Generate graph
  std::cout << "Generating graph..." << std::endl;
  auto g = generate_krongraph(prop.scale, prop.degree);

  // Print stats for graph
  auto num_nodes = g->num_idx_nodes;
  auto num_edges = g->index[g->num_idx_nodes];
  std::cout << "Graph name: " << prop.ofname << std::endl
            << "  # of nodes: " << num_nodes << std::endl
            << "  # of edges: " << num_edges << std::endl
            << "  avg degree: " << (double)num_edges / num_nodes << std::endl;

  // Write out graph
  std::cout << "Writing graph to file..." << std::endl;
  std::ofstream ofs(prop.ofname, std::ofstream::out);
  ofs << *g;
  ofs.close();

  return EXIT_SUCCESS;
}
