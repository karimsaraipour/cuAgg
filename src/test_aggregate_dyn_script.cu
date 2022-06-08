#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <math.h>
#include <utility>

#include "aggregate.cuh"
#include "cuda.cuh"
#include "generator.h"
#include "graph.h"

#ifndef SCALE
#define SCALE 10
#endif

#ifndef DEGREE
#define DEGREE 10
#endif

#ifndef WARP
#define WARP 4
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

bool feq(float f1, float f2) {
    return fabs(f1 - f2) < 0.001;
}

void aggregate_cpu_oracle(const GraphPtr g, const FeatureVec &in_features,
                          FeatureVec &out_features, int num_features) {
    FeatureVec node_features(num_features);

    for (NodeT v = 0; v < g->num_nodes; v++) {
        // Reset node features
        for (IndexT f = 0; f < num_features; f++)
            node_features[f] = in_features[v * num_features + f];

        // Aggregate features
        for (IndexT i = g->index[v]; i < g->index[v + 1]; i++) {
            NodeT u = g->neighbors[i];
            for (IndexT f = 0; f < num_features; f++)
                node_features[f] += in_features[u * num_features + f];
        }

        // Write out
        for (IndexT f = 0; f < num_features; f++)
            out_features[v * num_features + f] = node_features[f];
    }
}

int main() {

    // constexpr int TEST_SCALE = 14;
    // constexpr int TEST_DEGREE = 10;
    constexpr IndexT TEST_NUM_FEATURES = 64;

    constexpr int BLOCK_DIM_X = 16;
    constexpr int BLOCK_DIM_Y = 32;

    // Generate graph
    auto g = generate_krongraph(SCALE, DEGREE);
    assert(g != nullptr && "graph is invalid");

    // Get CPU oracle (single-threaded)
    auto features = generate_features(g->num_nodes, TEST_NUM_FEATURES);
    assert(!features.empty() && "features are empty");
    FeatureVec oracle_features(features.size());

    aggregate_cpu_oracle(g, features, oracle_features, TEST_NUM_FEATURES);

    // Get GPU aggregated features
    IndexT *cu_index;
    NodeT *cu_neighbors;
    FeatureT *cu_in_features;
    FeatureT *cu_out_features, *out_features;

    size_t size_index = g->index.size() * sizeof(IndexT);
    size_t size_neighbors = g->neighbors.size() * sizeof(NodeT);
    size_t size_features = features.size() * sizeof(FeatureT);
    size_t size_tile = 32;
    size_t num_streams = g->num_nodes / size_tile;
    // size_t size_stream = size_tile * (sizeof(IndexT) + sizeof(NodeT) + sizeof(FeatureT) + TEST_NUM_FEATURES);
    // size_t streamBytes = size_stream * sizeof(float);

    out_features = (FeatureT *)malloc(size_features);
    memset(out_features, 0, size_features);

    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i)
        CUDA_ERRCHK(cudaStreamCreate(&streams[i]));

    CUDA_ERRCHK(cudaMalloc((void **)&cu_index, size_index));
    CUDA_ERRCHK(cudaMalloc((void **)&cu_neighbors, size_neighbors));
    CUDA_ERRCHK(cudaMalloc((void **)&cu_in_features, size_features));
    CUDA_ERRCHK(cudaMalloc((void **)&cu_out_features, size_features));
    // CUDA_ERRCHK(cudaMemcpy(cu_index, g->index.data(), size_index,
    //                        cudaMemcpyHostToDevice));
    // CUDA_ERRCHK(cudaMemcpy(cu_neighbors, g->neighbors.data(), size_neighbors,
    //                        cudaMemcpyHostToDevice));
    // CUDA_ERRCHK(cudaMemcpy(cu_in_features, features.data(), size_features,
    //                        cudaMemcpyHostToDevice));
    // CUDA_ERRCHK(cudaMemset(cu_out_features, 0, size_features));

    dim3 dim_block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 dim_grid((g->num_nodes + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                  (TEST_NUM_FEATURES + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    auto start = std::chrono::high_resolution_clock::now();
    // aggregate_dyn<<<64, WARP * WARP_SIZE>>>(cu_index, cu_neighbors,
    //                                         cu_in_features, cu_out_features,
    //                                         g->num_nodes, TEST_NUM_FEATURES);

    for (int i = 0; i < num_streams; ++i) {
        // CUDA_ERRCHK(cudaMemcpyAsync(&d_a[offset], &a[offset],
        //                             streamBytes, cudaMemcpyHostToDevice,
        //                             streams[i]));

        CUDA_ERRCHK(cudaMemcpyAsync(cu_index + i * size_tile * sizeof(IndexT), g->index.data() + i * size_tile * sizeof(IndexT), size_tile * sizeof(IndexT), cudaMemcpyHostToDevice, streams[i]));
        CUDA_ERRCHK(cudaMemcpyAsync(cu_neighbors + i * size_tile * sizeof(NodeT), g->neighbors.data() + i * size_tile * sizeof(NodeT), size_tile * sizeof(NodeT), cudaMemcpyHostToDevice, streams[i]));
        CUDA_ERRCHK(cudaMemcpyAsync(cu_in_features + i * size_tile * sizeof(FeatureT), features.data() + i * size_tile * sizeof(FeatureT), size_tile * sizeof(FeatureT), cudaMemcpyHostToDevice, streams[i]));
        CUDA_ERRCHK(cudaMemcpyAsync(cu_out_features + i * size_tile * sizeof(FeatureT), out_features + i * size_tile * sizeof(FeatureT), size_tile * sizeof(FeatureT), cudaMemcpyHostToDevice, streams[i]));
    }
    for (int i = 0; i < num_streams; ++i) {
        //   int offset = i * size_stream;
        aggregate_dyn<<<64, WARP * WARP_SIZE, 0, streams[i]>>>(
            cu_index + i * size_tile * sizeof(IndexT),
            cu_neighbors + i * size_tile * sizeof(NodeT),
            cu_in_features + i * size_tile * sizeof(FeatureT),
            cu_out_features + i * size_tile * sizeof(FeatureT),
            g->num_nodes, TEST_NUM_FEATURES);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto kernel_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e9;
    fprintf(stdout, "Kernel execution time for %d warps: %f s\n", WARP, kernel_time);

    // Copy results to CPU memory
    FeatureT *test_features = new FeatureT[features.size() * sizeof(FeatureT)];
    // FeatureT *test_features = new FeatureT[features.size()];
    // CUDA_ERRCHK(cudaMemcpy(test_features, cu_out_features, size_features,
    //                        cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_streams; ++i) {
        printf("HERE%d\n", i);
        CUDA_ERRCHK(cudaMemcpyAsync(test_features + i * size_tile * sizeof(FeatureT), cu_out_features + i * size_tile * sizeof(FeatureT), size_tile * TEST_NUM_FEATURES * sizeof(FeatureT),
                                    cudaMemcpyDeviceToHost, streams[i]));
    }
    for (int i = 0; i < num_streams; ++i)
        CUDA_ERRCHK(cudaStreamDestroy(streams[i]));

    for (size_t i = 0; i < features.size(); i++)
        assert(feq(test_features[i], oracle_features[i]) && "features don't match");

    delete[] test_features;

    return EXIT_SUCCESS;
}
