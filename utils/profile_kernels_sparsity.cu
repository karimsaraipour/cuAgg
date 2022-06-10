#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "../src/cuda.cuh"
#include "../src/graph/generator.h"
#include "../src/graph/graph.h"
#include "../src/graph/partition.h"
/*#include "../src/kernels/aggregate.cuh"*/
/*#include "../src/kernels/aggregate_templated.cuh"*/

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <num_features>" << std::endl;
    return EXIT_FAILURE;
  }

  /*// Initialzie timing objects*/
  /*cudaEvent_t start;*/
  /*cudaEvent_t stop;*/
  /*cudaEventCreate(&start);*/
  /*cudaEventCreate(&stop);*/

  /*// Define kernels*/
  /*auto agg_dyn = [&start, &stop](const int num_warps) {*/
  /*return [&start, &stop, num_warps](*/
  /*const IndexT *const index, const NodeT *const neighbors,*/
  /*const FeatureT *const in_features, FeatureT *const out_features,*/
  /*const NodeT num_nodes, const IndexT num_features) -> void {*/
  /*cudaEventRecord(start);*/
  /*aggregate_dyn<<<num_nodes, num_warps * 32>>>(*/
  /*index, neighbors, in_features, out_features, num_nodes, num_features);*/
  /*cudaEventRecord(stop);*/
  /*};*/
  /*};*/

  /*auto agg_dyn_sm = [&start, &stop](const int num_warps) {*/
  /*return [&start, &stop, num_warps](*/
  /*const IndexT *const index, const NodeT *const neighbors,*/
  /*const FeatureT *const in_features, FeatureT *const out_features,*/
  /*const NodeT num_nodes, const IndexT num_features) -> void {*/
  /*cudaEventRecord(start);*/
  /*aggregate_dyn_sm<1024><<<num_nodes, num_warps * 32>>>(*/
  /*index, neighbors, in_features, out_features, num_nodes, num_features);*/
  /*cudaEventRecord(stop);*/
  /*};*/
  /*};*/

  /*auto agg_dyn_low = agg_dyn(1);*/
  /*auto agg_dyn_medium = agg_dyn(16);*/
  /*auto agg_dyn_high = agg_dyn(32);*/

  IndexT num_features = atoi(argv[1]);

  std::vector<float> sparsities = {0.05, 0.1, 0.15, 0.2, 0.3, 0.4,
                                   0.5,  0.6, 0.7,  0.8, 0.9};

  /*NodeT num_nodes = get_square_tile_size(num_features, 1, 1);*/
  NodeT num_nodes = 1 << 7;

  float sparsity = 0.5f;
  auto g = generate_graph_sparsity(num_nodes, sparsity);

  /*auto partitions = partition_square_tile(g, num_nodes);*/

  /*auto features = generate_features(num_nodes, num_features);*/
  /*FeatureT *dummy_features = new FeatureT[features.size()];*/
  /*std::cout << "features generated" << std::endl;*/

  /*aggregate_double_buffer_naive(partitions, 1, features, dummy_features,*/
  /*num_features, num_nodes, agg_dyn_high, 1);*/

  /*CUDA_ERRCHK(cudaEventSynchronize(stop));*/
  /*float elapsed;*/
  /*CUDA_ERRCHK(cudaEventElapsedTime(&elapsed, start, stop));*/
  /*std::cout << "Runtime: " << elapsed << " ms" << std::endl;*/

  return EXIT_SUCCESS;
}
