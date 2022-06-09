#include <algorithm>
#include <cstdlib>
#include <fstream>

#include <pngwriter.h>

#include "../src/graph/graph.h"

int main(int argc, char *argv[]) {
  if (argc == 1) {
    std::cerr << "Usage: " << argv[0] << " <graph_file> [<graph_file> ...]"
              << std::endl;
    return EXIT_FAILURE;
  }

  for (int argi = 1; argi < argc; argi++) {
    std::string fname_graph = argv[argi];

    std::ifstream ifs(fname_graph);
    auto g = GraphPtr(new Graph());
    ifs >> *g;

    std::string fname_png = fname_graph;
    {
      std::string ext = ".g";
      auto start = fname_png.find(ext);
      if (start != std::string::npos)
        fname_png.replace(start, ext.length(), ".png");
    }

    // Visualize graph
    float color_edge[3] = {1.0f, 1.0f, 1.0f};

    // rows = in-neighbors
    // cols = destinations
    // img.plot (column, row) 1-indexed
    // origin = bottom-left
    auto rows = g->num_neighbors;
    auto cols = g->num_idx_nodes;
    pngwriter img(cols, rows, 0.0f, fname_png.c_str());

#define SET_PIXEL(img, r, c, color)                                            \
  img.plot((c) + 1, rows - (r), color[0], color[1], color[2]);

    // Draw edges
#pragma omp parallel for
    for (NodeT v = 0; v < g->num_idx_nodes; v++) {
      for (IndexT i = g->index[v]; i < g->index[v + 1]; i++) {
        NodeT u = g->neighbors[i];
        SET_PIXEL(img, u, v, color_edge);
      }
    }

#undef SET_PIXEL

    img.close();
  }

  return EXIT_SUCCESS;
}
