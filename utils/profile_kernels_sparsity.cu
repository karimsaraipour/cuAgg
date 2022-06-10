#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "yaml-cpp/yaml.h"

#include "../src/cuda.cuh"
#include "../src/graph/generator.h"
#include "../src/graph/graph.h"
#include "../src/graph/partition.h"
#include "../src/kernels/aggregate.cuh"
#include "../src/kernels/aggregate_templated.cuh"

/** Assumes only one partition. */
class CustomDoubleBufferKernel {
public:
  CustomDoubleBufferKernel(const size_t tile_size_, const IndexT num_features_,
                           const size_t size_neighbors_ = 0)
      : tile_size(tile_size_), num_features(num_features_) {
    size_index = (tile_size + 1) * sizeof(IndexT);
    size_t w_tile_size = tile_size;
    size_t size_neighbors =
        ((size_neighbors_ == 0) ? w_tile_size * w_tile_size : size_neighbors_) *
        sizeof(NodeT);
    feature_count = tile_size * num_features;
    size_features = feature_count * sizeof(FeatureT);

    // Allocate arrays
    CUDA_ERRCHK(cudaMalloc((void **)&cu_index, size_index));
    CUDA_ERRCHK(cudaMalloc((void **)&cu_neighbors, size_neighbors));
    CUDA_ERRCHK(cudaMalloc((void **)&cu_in_features, size_features));
    CUDA_ERRCHK(cudaMalloc((void **)&cu_out_features, size_features));
  }

  ~CustomDoubleBufferKernel() {
    // Free memory
    CUDA_ERRCHK(cudaFree(cu_index));
    CUDA_ERRCHK(cudaFree(cu_neighbors));
    CUDA_ERRCHK(cudaFree(cu_in_features));
    CUDA_ERRCHK(cudaFree(cu_out_features));
  }

  void load_graph(const GraphPtr g) {
    size_t size_g_nghs = g->index.get()[g->num_idx_nodes] * sizeof(NodeT);
    CUDA_ERRCHK(cudaMemcpyAsync(cu_index, g->index.get(), size_index,
                                cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpyAsync(cu_neighbors, g->neighbors.get(), size_g_nghs,
                                cudaMemcpyHostToDevice));
  }

  void load_features(const FeatureVec &in_features) {
    CUDA_ERRCHK(cudaMemcpyAsync(cu_in_features, in_features.get(),
                                size_features, cudaMemcpyHostToDevice));
  }

  void execute_kernel(AggregateFunc kernel) {
    // Assumes graph & input features are already loaded
    CUDA_ERRCHK(cudaMemsetAsync(cu_out_features, 0, size_features));

    kernel(cu_index, cu_neighbors, cu_in_features, cu_out_features, tile_size,
           num_features);
    CUDA_ERRCHK(cudaDeviceSynchronize());
  }

private:
  IndexT *cu_index;
  NodeT *cu_neighbors;
  FeatureT *cu_in_features;
  FeatureT *cu_out_features;

  IndexT num_features;
  IndexT feature_count;
  size_t size_index;
  size_t size_features;
  size_t tile_size;
};

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <num_features>" << std::endl;
    return EXIT_FAILURE;
  }

  // Initialzie timing objects
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Define kernels
  auto agg_naive =
      [&start, &stop](const IndexT *const index, const NodeT *const neighbors,
                      const FeatureT *const in_features,
                      FeatureT *const out_features, const NodeT num_nodes,
                      const IndexT num_features) -> void {
    constexpr int BLOCK_DIM_X = 16;
    constexpr int BLOCK_DIM_Y = 32;

    dim3 dim_block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 dim_grid((num_nodes + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                  (num_features + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    cudaEventRecord(start);
    aggregate_naive<<<dim_grid, dim_block>>>(
        index, neighbors, in_features, out_features, num_nodes, num_features);
    cudaEventRecord(stop);
  };

  constexpr int tb_reuse = 4;
  auto agg_dyn = [&start, &stop, tb_reuse](const int num_warps) {
    return [&start, &stop, num_warps, tb_reuse](
               const IndexT *const index, const NodeT *const neighbors,
               const FeatureT *const in_features, FeatureT *const out_features,
               const NodeT num_nodes, const IndexT num_features) -> void {
      cudaEventRecord(start);
      aggregate_dyn<<<num_nodes / tb_reuse, num_warps * 32>>>(
          index, neighbors, in_features, out_features, num_nodes, num_features);
      cudaEventRecord(stop);
    };
  };

  constexpr int tile_size = 1024;
  auto agg_dyn_sm = [&start, &stop, tb_reuse, tile_size](const int num_warps) {
    return [&start, &stop, num_warps, tb_reuse, tile_size](
               const IndexT *const index, const NodeT *const neighbors,
               const FeatureT *const in_features, FeatureT *const out_features,
               const NodeT num_nodes, const IndexT num_features) -> void {
      cudaEventRecord(start);
      aggregate_dyn_sm<tile_size><<<num_nodes / tb_reuse, num_warps * 32>>>(
          index, neighbors, in_features, out_features, num_nodes, num_features);
      cudaEventRecord(stop);
    };
  };

  auto agg_dyn_rf = [&start, &stop, tb_reuse, tile_size](const int num_warps) {
    return [&start, &stop, num_warps, tb_reuse, tile_size](
               const IndexT *const index, const NodeT *const neighbors,
               const FeatureT *const in_features, FeatureT *const out_features,
               const NodeT num_nodes, const IndexT num_features) -> void {
      cudaEventRecord(start);
      aggregate_dyn_rf<tile_size><<<num_nodes / tb_reuse, num_warps * 32>>>(
          index, neighbors, in_features, out_features, num_nodes, num_features);
      cudaEventRecord(stop);
    };
  };

  auto agg_dyn_sm_rf = [&start, &stop, tb_reuse,
                        tile_size](const int num_warps) {
    return [&start, &stop, num_warps, tb_reuse, tile_size](
               const IndexT *const index, const NodeT *const neighbors,
               const FeatureT *const in_features, FeatureT *const out_features,
               const NodeT num_nodes, const IndexT num_features) -> void {
      cudaEventRecord(start);
      aggregate_dyn_rf<tile_size><<<num_nodes / tb_reuse, num_warps * 32>>>(
          index, neighbors, in_features, out_features, num_nodes, num_features);
      cudaEventRecord(stop);
    };
  };

  using KernelNamePair = std::pair<AggregateFunc, std::string>;
  std::vector<KernelNamePair> kernels = {
      std::make_pair(agg_naive, "Aggregate Naive"),
      std::make_pair(agg_dyn(1), "Aggregate Low"),
      std::make_pair(agg_dyn(16), "Aggregate Medium"),
      std::make_pair(agg_dyn(32), "Aggregate High"),
      std::make_pair(agg_dyn_sm(1), "Aggregate SM Low"),
      std::make_pair(agg_dyn_sm(16), "Aggregate SM Medium"),
      std::make_pair(agg_dyn_sm(32), "Aggregate SM High"),
      /*std::make_pair(agg_dyn_rf(1), "Aggregate RF Low"),*/
      /*std::make_pair(agg_dyn_rf(16), "Aggregate RF Medium"),*/
      /*std::make_pair(agg_dyn_rf(32), "Aggregate RF High"),*/
      /*std::make_pair(agg_dyn_sm_rf(1), "Aggregate RF+SM Low"),*/
      /*std::make_pair(agg_dyn_sm_rf(16), "Aggregate RF+SM Medium"),*/
      /*std::make_pair(agg_dyn_sm_rf(32), "Aggregate RF+SM High"),*/
  };

  IndexT num_features = atoi(argv[1]);

  NodeT num_nodes = get_square_tile_size(num_features, 2, 1);
  std::cout << "Num nodes: " << num_nodes << std::endl;

  auto features = generate_features(num_nodes, num_features);
  std::cout << "Generated features" << std::endl;

  YAML::Node results;
  results["prop"]["num_nodes"] = num_nodes;

  std::vector<float> sparsities = {1.0f / num_nodes,    // Degree 1
                                   2.0f / num_nodes,    // Degree 2
                                   5.0f / num_nodes,    // Degree 5
                                   10.f / num_nodes,    // Degree 10
                                   20.0f / num_nodes,   // Degree 20,
                                   50.0f / num_nodes,   // Degree 50
                                   100.0f / num_nodes,  // Degree 100
                                   200.0f / num_nodes,  // Degree 200
                                   500.0f / num_nodes,  // Degree 500
                                   1000.0f / num_nodes, // Degree 1000
                                   0.05f,
                                   0.1f,
                                   0.25f,
                                   0.5f,
                                   0.75f,
                                   1.0f};

  CustomDoubleBufferKernel runner(num_nodes, num_features);
  runner.load_features(features);

  // Profile
  for (auto i = 0; i < sparsities.size(); i++) {
    float sparsity = sparsities[i];

    auto g = generate_graph_sparsity(num_nodes, sparsity);
    std::cout << "Generated graph" << std::endl;

    PartitionVec partitions(1);
    partitions[0].subgraph = g;
    partitions[0].idx_map = NodeMapping::new_affine(0);
    partitions[0].ngh_map = NodeMapping::new_affine(0);

    runner.load_graph(g);

    results["results"][i]["prop"]["sparsity"] = sparsity;

    std::cout << "Profiling for sparsity: " << sparsity << std::endl;
    for (const auto &ker_name_pair : kernels) {
      size_t feature_count = num_nodes * num_features;
      auto dummy_features = FeatureVector::create(feature_count);

      auto kernel_func = ker_name_pair.first;
      auto kernel_name = ker_name_pair.second;

      runner.execute_kernel(kernel_func);

      CUDA_ERRCHK(cudaEventSynchronize(stop));
      float elapsed;
      CUDA_ERRCHK(cudaEventElapsedTime(&elapsed, start, stop));
      std::cout << kernel_name << " runtime: " << elapsed << " ms" << std::endl;

      YAML::Node kernel_result;
      kernel_result["name"] = kernel_name;
      kernel_result["runtime"] = elapsed;
      results["results"][i]["runs"].push_back(kernel_result);
    }
  }

  std::cout << results << std::endl;

  std::stringstream ss;
  ss << "profile_sparsity_" << num_nodes << "_" << num_features << ".yaml";

  std::ofstream ofs(ss.str(), std::ofstream::out);
  ofs << results;
  ofs.close();

  return EXIT_SUCCESS;
}
