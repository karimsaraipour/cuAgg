// #include "aggregate.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

// define number of nodes/vertices
// define number of feature vector dimensions
#ifndef NNODES
#define NNODES 10
#endif

#ifndef FEATURE_DIM
#define FEATURE_DIM 16
#endif

// typedef adjacency and feature matrix datatype
typedef bool ADJ_DTYPE;
typedef float FEATURE_DTYPE;

// block dimensions
#ifndef BLOCK_DIM_X
#define BLOCK_DIM_X 16
#endif

#ifndef BLOCK_DIM_Y
#define BLOCK_DIM_Y 32
#endif


// cuda error check
#define cudaErrCheck(__status) { cudaAssert(__status, __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t __status, const char *file, int line) {
    if (__status != cudaSuccess) {
        fprintf(stderr, "[%s, %d]: %s - Terminating...\n", file, line, cudaGetErrorString(__status));
        exit(EXIT_FAILURE);
    }
}

// malloc error checking on host
#define hostMallocErrCheck(ptr) { mallocErrCheck(ptr, __FILE__, __LINE__); }
inline void mallocErrCheck(void *ptr, const char *file, int line) {
    if (ptr == NULL) {
        fprintf(stderr, "[%s, %d]: Host memory allocation failed! Terminating...\n", file, line);
        exit(EXIT_FAILURE);
    }
}

// CSR struct
struct CSR {

    int *col_idx, *row_idx;
    int col_idx_len, row_idx_len;

    CSR() = default;
    CSR(ADJ_DTYPE *adj_matrix);
    ~CSR() {
        free(col_idx);
        free(row_idx);
    }

    void dump();
    
};

CSR::CSR(ADJ_DTYPE *adj_matrix) {

    row_idx_len = NNODES+1;
    row_idx = static_cast<int *>(malloc(row_idx_len * sizeof(int)));
    hostMallocErrCheck(row_idx);

    col_idx_len = 0;
    for (int r = 0; r < NNODES; ++r) {
        for (int c = 0; c < NNODES; ++c) {
            if (adj_matrix[r * NNODES + c] != 0) ++col_idx_len;
        }
    }

    col_idx = static_cast<int *>(malloc(col_idx_len * sizeof(int)));
    hostMallocErrCheck(col_idx);

    int nz_idx = 0;
    for (int r = 0; r < NNODES; ++r) {
        row_idx[r] = nz_idx;
        for (int c = 0; c < NNODES; ++c) {
            if (adj_matrix[r * NNODES + c] != 0) {
                col_idx[nz_idx] = c;
                ++nz_idx;
            }
        }
    }
    row_idx[row_idx_len - 1] = col_idx_len;

}

// CSR dump
void CSR::dump() {

    printf("COL INDEX: ");
    for (int i = 0; i < col_idx_len; ++i) {
        printf("%d ", col_idx[i]);
    }
    printf("\n");

    printf("ROW INDEX: ");
    for (int i = 0; i < row_idx_len; ++i) {
        printf("%d ", row_idx[i]);
    }
    printf("\n");

    printf("COL INDEX LENGTH: %d\n", col_idx_len);
    printf("ROW INDEX LENGTH: %d\n", row_idx_len);
    printf("\n");

}

// init adj matrix
void init_adj_matrix(ADJ_DTYPE *adj_matrix) {

    srand(time(NULL));

    for (int r = 0; r < NNODES; ++r) {
        for (int c = 0; c < NNODES; ++c) {
            if (r == c) continue; // skip a self edge. mnight want to change this later
            adj_matrix[r * NNODES + c] = 
                static_cast<ADJ_DTYPE>(roundf(static_cast<float>(rand())/static_cast<float>(RAND_MAX)));
        }
    }

}

// init feature matrix
void init_feature_matrix(FEATURE_DTYPE *feature_matrix) {

    srand(time(NULL));

    for (int v = 0; v < NNODES; ++v) {
        for (int d = 0; d < FEATURE_DIM; ++d) {
            feature_matrix[v * FEATURE_DIM + d] = 
                static_cast<FEATURE_DTYPE>(rand())/static_cast<FEATURE_DTYPE>(RAND_MAX) - 0.5f;
        }
    }

}

// dump adjacency matrix
void dump_raw_adjacency(ADJ_DTYPE *adj_matrix) {

    printf("ADJ MATRIX:\n");
    for (int r = 0; r < NNODES; ++r) {
        for (int c = 0; c < NNODES; ++c) {
            printf("%d ", adj_matrix[r * NNODES + c]);
        }
        printf("\n");
    }
    printf("\n");

}

// dump features
void dump_features(FEATURE_DTYPE *features) {

    printf("FEATURES:\n");
    for (int v = 0; v < NNODES; ++v) {
        for (int d = 0; d < FEATURE_DIM; ++d) {
            printf("%.3f ", features[v * FEATURE_DIM + d]);
        }
        printf("\n");
    }
    printf("\n");

}

// device feature aggregation kernel
__global__ void aggregate(int *__csr_col, int *__csr_row, FEATURE_DTYPE *__src, FEATURE_DTYPE *__dest) {

    // copy src to dest before aggregating neighbors
    for (int vertex = threadIdx.x + blockIdx.x * blockDim.x;
        vertex < NNODES; vertex += blockDim.x * gridDim.x) {
            int left = __csr_row[vertex];
            int right = __csr_row[vertex + 1];
            for (int fdim = threadIdx.y + blockIdx.y * blockDim.y;
                fdim < FEATURE_DIM; fdim += blockDim.y * gridDim.y) {
                    __dest[vertex * FEATURE_DIM + fdim] = __src[vertex * FEATURE_DIM + fdim];
            }
            for (int nbr = left; nbr < right; nbr++) {
                int neighbor = __csr_col[nbr];
                for (int fdim = threadIdx.y + blockIdx.y * blockDim.y;
                    fdim < FEATURE_DIM; fdim += blockDim.y * gridDim.y) {
                        __dest[vertex * FEATURE_DIM + fdim] += __src[neighbor * FEATURE_DIM + fdim];
                }
            }
    }

}

int main(void) {

    // host pointers
    ADJ_DTYPE *adj_matrix;
    FEATURE_DTYPE *feature_matrix, *aggregated_features;

    // sizes of adjacency and feature matrices
    size_t size_adj = 1L*NNODES*NNODES,
            size_fm = 1L*NNODES*FEATURE_DIM;

    // allocate memory on host
    adj_matrix = static_cast<ADJ_DTYPE *>(malloc(size_adj * sizeof(ADJ_DTYPE)));
    hostMallocErrCheck(adj_matrix);

    feature_matrix = static_cast<FEATURE_DTYPE *>(malloc(size_fm * sizeof(FEATURE_DTYPE)));
    hostMallocErrCheck(feature_matrix);

    aggregated_features = static_cast<FEATURE_DTYPE *>(malloc(size_fm * sizeof(FEATURE_DTYPE)));
    hostMallocErrCheck(aggregated_features);

    // initialize adjacency matrix
    init_adj_matrix(adj_matrix);

    // intialize feature matrix
    init_feature_matrix(feature_matrix);

    // obtain CSR form of adjacency matrix
    CSR csr(adj_matrix);

    // allocate memory on device
    FEATURE_DTYPE *d_feature_matrix, *d_aggregated_features;
    int *d_col_idx, *d_row_idx;
    cudaErrCheck( cudaMalloc((void **)&d_feature_matrix, size_fm * sizeof(FEATURE_DTYPE)) );
    cudaErrCheck( cudaMalloc((void **)&d_aggregated_features, size_fm * sizeof(FEATURE_DTYPE)) );
    cudaErrCheck( cudaMalloc((void **)&d_col_idx, csr.col_idx_len * sizeof(int)) );
    cudaErrCheck( cudaMalloc((void **)&d_row_idx, csr.row_idx_len * sizeof(int)) );

    // copy features and CSR to device
    cudaErrCheck( cudaMemcpy(d_feature_matrix, feature_matrix, size_fm * sizeof(FEATURE_DTYPE), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(d_col_idx, csr.col_idx, csr.col_idx_len * sizeof(int), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(d_row_idx, csr.row_idx, csr.row_idx_len * sizeof(int), cudaMemcpyHostToDevice) );

    // determine the size of a block and the grid
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 dimGrid((NNODES + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (FEATURE_DIM + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    // invoke kernel
    aggregate<<<dimGrid, dimBlock>>>(d_col_idx, d_row_idx, d_feature_matrix, d_aggregated_features);
    cudaErrCheck( cudaPeekAtLastError() );

    // obtain aggregated features from device
    cudaErrCheck( cudaMemcpy(aggregated_features, d_aggregated_features, size_fm * sizeof(FEATURE_DTYPE), cudaMemcpyDeviceToHost) );

    // DEBUG
    // dump_raw_adjacency(adj_matrix);
    // printf("ORIGINAL ");
    // dump_features(feature_matrix);
    // printf("AGGREGATED ");
    // dump_features(aggregated_features);

    // free device memory
    cudaFree(d_feature_matrix);
    cudaFree(d_aggregated_features);
    cudaFree(d_col_idx);
    cudaFree(d_row_idx);

    // free memory on host
    free(adj_matrix);
    free(feature_matrix);
    free(aggregated_features);

    return 0;
}