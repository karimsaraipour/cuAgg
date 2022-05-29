// #include "aggregate.cuh"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

// notes
// build a matrix of feature vectors
// one feature vector per node
#ifndef NNODES
#define NNODES 10
#endif

#ifndef FEATURE_DIM
#define FEATURE_DIM 32
#endif

// cuda error checking
#define cudaErrCheck(__status) { cudaAssert(__status, __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t __status, const char *file, int line) {
    if (__status != cudaSuccess) {
        fprintf(stderr, "[%s, %d]: %s - Terminating...\n", file, line, cudaGetErrorString(__status));
        exit(EXIT_FAILURE);
    }
}

// malloc error check
#define mallocErrCheck(ptr) { memAssert(ptr, __FILE__, __LINE__); }
inline void memAssert(void *ptr, const char *file, int line) {
    if (ptr == NULL) {
        fprintf(stderr, "[%s, %s]: Host memory allocation failed. Terminating...\n", file, line);
        exit(EXIT_FAILURE);
    }
}

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
    mallocErrCheck(row_idx);

    // nnz
    col_idx_len = 0;
    for (int r = 0; r < NNODES; ++r)
        for (int c = 0; c < NNODES; ++c)
            if (adj_matrix[r * NNODES + c] != 0)
                ++col_idx_len;

    col_idx = static_cast<int *>(malloc(col_idx_len * sizeof(int)));
    mallocErrCheck(col_idx);

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

    printf("CSR:\n");
    printf("COL INDEX ARRAY: ");
    for (int c = 0; c < col_idx_len; ++c)
        printf("%d ", col_idx[c]);
    printf("\n");

    printf("ROW INDEX ARRAY: ");
    for (int r = 0; r < row_idx_len; ++r)
        printf("%d ", row_idx[r]);
    printf("\n");

    printf("COL INDEX LENGTH: %d\n", col_idx_len);
    printf("ROW INDEX LENGTH: %d\n", row_idx_len);
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
    mallocErrCheck(col_idx);

    row_idx_len = 0;
    for (int r = 0; r < NNODES; ++r)
        for (int c = 0; c < NNODES; ++c)
            if (adj_matrix[r * NNODES + c] != 0)
                ++row_idx_len;

    row_idx = static_cast<int *>(malloc(row_idx_len * sizeof(int)));
    mallocErrCheck(row_idx);

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

    printf("CSC:\n");
    printf("COL INDEX ARRAY: ");
    for (int c = 0; c < col_idx_len; ++c)
        printf("%d ", col_idx[c]);
    printf("\n");

    printf("ROW INDEX ARRAY: ");
    for (int r = 0; r < row_idx_len; ++r)
        printf("%d ", row_idx[r]);
    printf("\n");

    printf("COL INDEX LENGTH: %d\n", col_idx_len);
    printf("ROW INDEX LENGTH: %d\n", row_idx_len);
}

// initialize adjacency matrix
void adj_matrix_init(ADJ_DTYPE *adj_matrix) {

    srand(time(NULL));

    for (int r = 0; r < NNODES; ++r)
        for (int c = 0; c < NNODES; ++c) {
            if (c == r) {
                adj_matrix[r * NNODES + c] = 0;
                continue;
            }
            adj_matrix[r * NNODES + c] = 
                static_cast<ADJ_DTYPE>(roundf(static_cast<float>(rand()) / static_cast<float>(RAND_MAX)));
        }
}

// initialize feature matrix
void feature_matrix_init(FEATURE_DTYPE *feature_matrix) {

    srand(time(NULL));

    for (int n = 0; n < NNODES; ++n)
        for (int d = 0; d < FEATURE_DIM; ++d)
            feature_matrix[n * FEATURE_DIM + d] =
                static_cast<FEATURE_DTYPE>(rand()) / static_cast<FEATURE_DTYPE>(RAND_MAX) - 0.5f;
}

// print adjacency matrix
void print_adj_matrix(ADJ_DTYPE *adj_matrix) {

    printf("ADJ MATRIX:\n");
    for (int r = 0; r < NNODES; ++r) {
        for (int c = 0; c < NNODES; ++c)
            printf("%d ", adj_matrix[r * NNODES + c]);
        printf("\n");
    }
}

// print feature matrix
void print_feature_matrix(FEATURE_DTYPE *feature_matrix) {

    printf("FEATURES:\n");
    for (int n = 0; n < NNODES; ++n) {
        for (int d = 0; d < FEATURE_DIM; ++d)
            printf("%.3f ", feature_matrix[n * FEATURE_DIM + d]);
        printf("\n");
    }
}

// assumes destination on left side of adj matrix
//
__device__ int *get_neighbors(int index, int *col_idx, int *row_idx, int &neighbor_length) {
    int num_neighbors = row_idx[index + 1] - row_idx[index];

    // sigh, this is a lot of mallocing
    // also need to error check a malloc call on device
    // TODO(maybe): fix this
    int *neighbors = static_cast<int *>(malloc(num_neighbors * sizeof(int)));
    if (neighbors == NULL) {
        printf("[%s, %d]: Malloc call on device returned NULL. Terminating...\n", __FILE__, __LINE__);
        assert(0);
    }

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
    mallocErrCheck(adj_matrix);

    // allocate memory for feature matrix
    original_feature_matrix = static_cast<FEATURE_DTYPE *>(malloc(size_fm * sizeof(FEATURE_DTYPE)));
    mallocErrCheck(original_feature_matrix);

    agg_feature_matrix = static_cast<FEATURE_DTYPE *>(malloc(size_fm * sizeof(FEATURE_DTYPE)));
    mallocErrCheck(agg_feature_matrix);

    // init adjacency matrix
    adj_matrix_init(adj_matrix);

    // init feature matrix
    feature_matrix_init(original_feature_matrix);

    // show adjacency matrix
    // print_adj_matrix(adj_matrix);

    // obtain CSR representation of adjacency matrix
    CSR csr(adj_matrix);
    //csr.dump();

    // device pointers
    FEATURE_DTYPE *d_src_matrix, *d_dest_matrix;
    int *d_col_idx, *d_row_idx;

    // allocate memory on device
    cudaErrCheck( cudaMalloc((void **)&d_src_matrix, size_fm * sizeof(FEATURE_DTYPE)) );
    cudaErrCheck( cudaMalloc((void **)&d_dest_matrix, size_fm * sizeof(FEATURE_DTYPE)) );

    cudaErrCheck( cudaMalloc((void **)&d_col_idx, csr.col_idx_len * sizeof(int)) );
    cudaErrCheck( cudaMalloc((void **)&d_row_idx, csr.row_idx_len * sizeof(int)) );

    // memcpy
    cudaErrCheck( cudaMemcpy(d_src_matrix, original_feature_matrix, size_fm * sizeof(FEATURE_DTYPE), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(d_dest_matrix, agg_feature_matrix, size_fm * sizeof(FEATURE_DTYPE), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(d_col_idx, csr.col_idx, csr.col_idx_len * sizeof(int), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(d_row_idx, csr.row_idx, csr.row_idx_len * sizeof(int), cudaMemcpyHostToDevice) );

    // block and grid sizes
    dim3 dimBlock(NNODES, FEATURE_DIM);

    // invoke kernel
    dummy_aggregate_kernel<<<1, dimBlock>>>(d_col_idx, d_row_idx, d_src_matrix, d_dest_matrix);
    cudaErrCheck( cudaPeekAtLastError() );

    // copy aggregate feature matrix back into cpu
    cudaErrCheck( cudaMemcpy(agg_feature_matrix, d_dest_matrix, size_fm * sizeof(FEATURE_DTYPE), cudaMemcpyDeviceToHost) );
    
    // show feature matrix
    // print_feature_matrix(original_feature_matrix);
    // show aggregated feature matrix
    // print_feature_matrix(agg_feature_matrix);

    // aggregate_sum(csr.col_idx, csr.row_idx, original_feature_matrix, agg_feature_matrix);

    // free device memory
    cudaFree(d_src_matrix);
    cudaFree(d_dest_matrix);
    cudaFree(d_col_idx);
    cudaFree(d_row_idx);
    
    // free adjacency matrix on exit
    free(adj_matrix);
    // free feature matrix
    free(original_feature_matrix);
    // free feature matrix
    free(agg_feature_matrix);

    return 0;
}