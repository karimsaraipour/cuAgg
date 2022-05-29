// #include "aggregate.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

// notes
// build a matrix of feature vectors
// one feature vector per node
#ifndef NNODES
#define NNODES 10
#endif

#ifndef FEATURE_DIM
#define FEATURE_DIM 16
#endif

// malloc error check
// #define // MALLOC_ERR_CHECK(pointer)                               \
//     if (pointer == NULL) {                                      \
//         printf("Error: [%s, %s] Memory allocation failed!\n");  \
//         return 1;                                               \
//     }

typedef bool ADJ_DTYPE;
typedef float FEATURE_DTYPE;

struct CSR {

    int *col_idx, *row_idx;
    int col_idx_len, row_idx_len = (NNODES + 1);

    CSR() = default;
    CSR(ADJ_DTYPE *adj_matrix);
    ~CSR() {
        free(col_idx);
        free(row_idx);
    }

    void dump();
};


// grid dimensions analogous to feature matrix dimensions (n x d)
// each of the n nodes has a d-dimensional feature vector



CSR::CSR(ADJ_DTYPE *adj_matrix) {

    row_idx = static_cast<int *>(malloc(row_idx_len * sizeof(int)));
    //// MALLOC_ERR_CHECK(row_idx);

    // nnz
    col_idx_len = 0;
    for (int r = 0; r < NNODES; ++r)
        for (int c = 0; c < NNODES; ++c)
            if (adj_matrix[r * NNODES + c] != 0)
                ++col_idx_len;

    col_idx = static_cast<int *>(malloc(col_idx_len * sizeof(int)));
    //// MALLOC_ERR_CHECK(col_idx);

    int c_idx = 0;
    for (int r = 0; r < NNODES; ++r) {
        row_idx[r] = c_idx;
        for (int c = 0; c < NNODES; ++c)
            if (adj_matrix[r * NNODES + c] != 0) {
                col_idx[c_idx] = c;
                ++c_idx;
            }
    }
    row_idx[row_idx_len - 1] = col_idx_len;
}

void CSR::dump() {

    std::cout << "CSR:" << std::endl;
    std::cout << "COL INDEX ARRAY: ";
    for (int c = 0; c < col_idx_len; ++c)
        std::cout << col_idx[c] << " ";
    std::cout << std::endl;

    std::cout << "ROW INDEX ARRAY: ";
    for (int r = 0; r < row_idx_len; ++r)
        std::cout << row_idx[r] << " ";
    std::cout << std::endl;

    std::cout << "COL INDEX LENGTH: " << col_idx_len << std::endl;
    std::cout << "ROW INDEX LENGTH: " << row_idx_len << std::endl;
}

struct CSC {

    int *col_idx, *row_idx;
    int col_idx_len = (NNODES + 1), row_idx_len;

    CSC() = default;
    CSC(ADJ_DTYPE *adj_matrix);
    ~CSC() {
        free(col_idx);
        free(row_idx);
    }

    void dump();
};

CSC::CSC(ADJ_DTYPE *adj_matrix) {

    col_idx = static_cast<int *>(malloc(col_idx_len * sizeof(int)));
    //// MALLOC_ERR_CHECK(col_idx);

    row_idx_len = 0;
    for (int r = 0; r < NNODES; ++r)
        for (int c = 0; c < NNODES; ++c)
            if (adj_matrix[r * NNODES + c] != 0)
                ++row_idx_len;

    row_idx = static_cast<int *>(malloc(row_idx_len * sizeof(int)));
    //// MALLOC_ERR_CHECK(col_idx);

    int r_idx = 0;
    for (int c = 0; c < NNODES; ++c) {
        col_idx[c] = r_idx;
        for (int r = 0; r < NNODES; ++r)
            if (adj_matrix[r * NNODES + c] != 0) {
                row_idx[r_idx] = r;
                ++r_idx;
            }
    }
    col_idx[col_idx_len - 1] = row_idx_len;
}

void CSC::dump() {

    std::cout << "CSC:" << std::endl;
    std::cout << "COL INDEX ARRAY: ";
    for (int c = 0; c < col_idx_len; ++c)
        std::cout << col_idx[c] << " ";
    std::cout << std::endl;

    std::cout << "ROW INDEX ARRAY: ";
    for (int r = 0; r < row_idx_len; ++r)
        std::cout << row_idx[r] << " ";
    std::cout << std::endl;

    std::cout << "COL INDEX LENGTH: " << col_idx_len << std::endl;
    std::cout << "ROW INDEX LENGTH: " << row_idx_len << std::endl;
}

// initialize adjacency matrix
void adj_matrix_init(ADJ_DTYPE *adj_matrix) {

    std::srand(std::time(nullptr));

    for (int r = 0; r < NNODES; ++r)
        for (int c = 0; c < NNODES; ++c) {
            if (c == r) {
                adj_matrix[r * NNODES + c] = 0;
                continue;
            }
            adj_matrix[r * NNODES + c] = static_cast<ADJ_DTYPE>(roundf(
                static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)));
        }
}

// initialize feature matrix
void feature_matrix_init(FEATURE_DTYPE *feature_matrix) {

    std::srand(std::time(nullptr));

    for (int n = 0; n < NNODES; ++n)
        for (int d = 0; d < FEATURE_DIM; ++d)
            feature_matrix[n * FEATURE_DIM + d] =
                static_cast<FEATURE_DTYPE>(std::rand()) / static_cast<FEATURE_DTYPE>(RAND_MAX) - 0.5f;
}

// print adjacency matrix
void print_adj_matrix(ADJ_DTYPE *adj_matrix) {

    std::cout << "ADJ MATRIX:" << std::endl;
    for (int r = 0; r < NNODES; ++r) {
        for (int c = 0; c < NNODES; ++c)
            std::cout << adj_matrix[r * NNODES + c] << " ";
        std::cout << std::endl;
    }
}

// print feature matrix
void print_feature_matrix(FEATURE_DTYPE *feature_matrix) {

    std::cout << "FEATURES:" << std::endl;
    for (int n = 0; n < NNODES; ++n) {
        for (int d = 0; d < FEATURE_DIM; ++d)
            std::cout << feature_matrix[n * FEATURE_DIM + d] << " ";
        std::cout << std::endl;
    }
}

// assumes destination on left side of adj matrix
//
__device__ int *get_neighbors(int index, int *col_idx, int *row_idx, int &neighbor_length) {
    int num_neighbors = row_idx[index + 1] - row_idx[index];

    int *neighbors = static_cast<int *>(malloc(num_neighbors * sizeof(int)));
    // MALLOC_ERR_CHECK(neighbors);

    int start = row_idx[index];
    for (int i = 0; i < num_neighbors; i++) {
        neighbors[i] = col_idx[start + i];
    }

    // change index to num_neighbors to know length of neighbors
    neighbor_length = num_neighbors;
    return neighbors;
}



__global__ void dummy_aggregate_kernel(int *col_idx, int *row_idx, FEATURE_DTYPE *src_features, FEATURE_DTYPE *dest_features) {

    // get copy of original feature matrix
    for (int vertex = threadIdx.x + blockIdx.x * blockDim.x;
        vertex < NNODES; vertex += blockDim.x * gridDim.x) { // grid striding
        for (int feature_dim = threadIdx.y + blockIdx.y * blockDim.y;
            feature_dim < FEATURE_DIM; feature_dim += blockDim.y * gridDim.y) {
            dest_features[vertex * FEATURE_DIM + feature_dim] = src_features[vertex * FEATURE_DIM + feature_dim];
        }
    }

    // aggregation
    for (int vertex = threadIdx.x + blockIdx.x * blockDim.x;
        vertex < NNODES; vertex += blockDim.x * gridDim.x) { // grid striding
        int degree;
        int *neighbors = get_neighbors(vertex, col_idx, row_idx, degree);

        for (int n = 0; n < degree; n++) {
            // add column values from feature matrix into target node's feature vector
            int neighbor = neighbors[n];
            for (int feature_dim = threadIdx.y + blockIdx.y * blockDim.y;
                feature_dim < FEATURE_DIM; feature_dim += blockDim.y * gridDim.y) {
                dest_features[vertex * FEATURE_DIM + feature_dim] += src_features[neighbor * FEATURE_DIM + feature_dim];
            }
        }
        __syncthreads();
        free(neighbors);
    }
}

int main(void) {
    // set display precision
    std::cout << std::fixed;
    std::cout << std::setprecision(4);

    // pointer to the adjacency matrix
    ADJ_DTYPE *adj_matrix;
    // pointer to feature vector matrix
    FEATURE_DTYPE *original_feature_matrix, *agg_feature_matrix;

    // size of the adjacency matrix
    size_t size_adj = 1L * NNODES * NNODES;
    // size of feature matrix
    size_t size_fm = 1L * NNODES * FEATURE_DIM;

    // allocate memory for the adjacency matrix
    adj_matrix = static_cast<ADJ_DTYPE *>(malloc(size_adj * sizeof(ADJ_DTYPE)));
    // MALLOC_ERR_CHECK(adj_matrix);

    // allocate memory for feature matrix
    original_feature_matrix = static_cast<FEATURE_DTYPE *>(malloc(size_fm * sizeof(FEATURE_DTYPE)));
    // MALLOC_ERR_CHECK(original_feature_matrix);

    agg_feature_matrix = static_cast<FEATURE_DTYPE *>(malloc(size_fm * sizeof(FEATURE_DTYPE)));
    // MALLOC_ERR_CHECK(agg_feature_matrix);

    // init adjacency matrix
    adj_matrix_init(adj_matrix);

    // init feature matrix
    feature_matrix_init(original_feature_matrix);

    // show adjacency matrix
    print_adj_matrix(adj_matrix);

    // obtain CSR representation of adjacency matrix
    CSR csr(adj_matrix);
    csr.dump();

    // obtain CSC representation of adjacency matrix
    // CSC csc(adj_matrix);
    // csc.dump();

    // device pointers
    FEATURE_DTYPE *d_src_matrix, *d_dest_matrix;
    int *d_col_idx, *d_row_idx;

    // allocate memory on device
    cudaMalloc((void **)&d_src_matrix, size_fm * sizeof(FEATURE_DTYPE));
    cudaMalloc((void **)&d_dest_matrix, size_fm * sizeof(FEATURE_DTYPE));

    cudaMalloc((void **)&d_col_idx, csr.col_idx_len * sizeof(int));
    cudaMalloc((void **)&d_row_idx, csr.row_idx_len * sizeof(int));

    // memcpy
    cudaMemcpy(d_src_matrix, original_feature_matrix, size_fm * sizeof(FEATURE_DTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest_matrix, agg_feature_matrix, size_fm * sizeof(FEATURE_DTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, csr.col_idx, csr.col_idx_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_idx, csr.row_idx, csr.row_idx_len * sizeof(int), cudaMemcpyHostToDevice);

    // block and grid sizes
    dim3 dimBlock(NNODES, FEATURE_DIM);

    // invoke kernel
    dummy_aggregate_kernel<<<1, dimBlock>>>(d_col_idx, d_row_idx, d_src_matrix, d_dest_matrix);

    // copy aggregate feature matrix back into cpu
    cudaMemcpy(agg_feature_matrix, d_dest_matrix, size_fm * sizeof(FEATURE_DTYPE), cudaMemcpyDeviceToHost);
    
    // show feature matrix
    print_feature_matrix(original_feature_matrix);

    // aggregate_sum(csr.col_idx, csr.row_idx, original_feature_matrix, agg_feature_matrix);

    // free device memory
    cudaFree(d_src_matrix);
    cudaFree(d_dest_matrix);
    cudaFree(d_col_idx);
    cudaFree(d_row_idx);

    print_feature_matrix(agg_feature_matrix);
    // free adjacency matrix on exit
    free(adj_matrix);
    // free feature matrix
    free(original_feature_matrix);
    // free feature matrix
    free(agg_feature_matrix);

    return 0;
}