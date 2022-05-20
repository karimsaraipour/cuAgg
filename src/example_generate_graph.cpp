#include <cstdlib>

#include "generator.h"

int main(int argc, char *argv[]) {
  int scale = 5;  // 2^{scale} number of nodes
  int degree = 3; // average degree
  auto g = generate_krongraph(scale, degree);

  return EXIT_SUCCESS;
}
