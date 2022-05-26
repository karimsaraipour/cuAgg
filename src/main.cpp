#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>

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
#define MALLOC_ERR_CHECK(pointer) if (pointer == nullptr) { \
    std::cerr<<"Error: "<<"["<<__FILE__<<", "<<__LINE__<<"]"<<" Memory allocation failed!"<<std::endl; \
    EXIT_FAILURE; \
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

CSR::CSR(ADJ_DTYPE *adj_matrix) {

    row_idx = static_cast <int *> (malloc(row_idx_len * sizeof(int))); 
    MALLOC_ERR_CHECK(row_idx);

    col_idx_len = 0;
    for (int r = 0; r < NNODES; ++r)
        for (int c = 0; c < NNODES; ++c)
            if (adj_matrix[r*NNODES + c] != 0) ++col_idx_len;

    col_idx = static_cast <int *> (malloc(col_idx_len * sizeof(int)));
    MALLOC_ERR_CHECK(col_idx);

    int c_idx = 0;
    for (int r = 0; r < NNODES; ++r) {
        row_idx[r] = c_idx;
        for (int c = 0; c < NNODES; ++c) 
            if (adj_matrix[r*NNODES + c] != 0) {
                col_idx[c_idx] = c;
                ++c_idx;
            }
    }
    row_idx[row_idx_len - 1] = col_idx_len;
        
}

void CSR::dump() {

    std::cout<<"CSR:"<<std::endl;
    std::cout<<"COL INDEX ARRAY: ";
    for (int c = 0; c < col_idx_len; ++c)
        std::cout<<col_idx[c]<<" ";
    std::cout<<std::endl;

    std::cout<<"ROW INDEX ARRAY: ";
    for (int r = 0; r < row_idx_len; ++r)
        std::cout<<row_idx[r]<<" ";
    std::cout<<std::endl;

    std::cout<<"COL INDEX LENGTH: "<<col_idx_len<<std::endl;
    std::cout<<"ROW INDEX LENGTH: "<<row_idx_len<<std::endl;

}


struct CSC {

    int *col_idx, *row_idx;
    int col_idx_len = (NNODES + 1), row_idx_len;

    CSC() = default;
    CSC(ADJ_DTYPE* adj_matrix);
    ~CSC() {
        free(col_idx);
	free(row_idx);
    }

    void dump();

};

CSC::CSC(ADJ_DTYPE* adj_matrix) {

    col_idx = static_cast <int *> (malloc(col_idx_len * sizeof(int))); 
    MALLOC_ERR_CHECK(col_idx);

    row_idx_len = 0;
    for (int r = 0; r < NNODES; ++r)
        for (int c = 0; c < NNODES; ++c)
            if (adj_matrix[r*NNODES + c] != 0) ++row_idx_len;

    row_idx = static_cast <int *> (malloc(row_idx_len * sizeof(int)));
    MALLOC_ERR_CHECK(col_idx);

    int r_idx = 0;
    for (int c = 0; c < NNODES; ++c) {
        col_idx[c]= r_idx;
        for (int r = 0; r < NNODES; ++r) 
            if (adj_matrix[r*NNODES + c] != 0) {
                row_idx[r_idx] = r;
                ++r_idx;
            }  
    }
    col_idx[col_idx_len - 1] = row_idx_len; 

}

void CSC::dump() {

    std::cout<<"CSC:"<<std::endl;
    std::cout<<"COL INDEX ARRAY: ";
    for (int c = 0; c < col_idx_len; ++c)
        std::cout<<col_idx[c]<<" ";
    std::cout<<std::endl;

    std::cout<<"ROW INDEX ARRAY: ";
    for (int r = 0; r < row_idx_len; ++r)
        std::cout<<row_idx[r]<<" ";
    std::cout<<std::endl;

    std::cout<<"COL INDEX LENGTH: "<<col_idx_len<<std::endl;
    std::cout<<"ROW INDEX LENGTH: "<<row_idx_len<<std::endl;

}

// initialize adjacency matrix
void adj_matrix_init(ADJ_DTYPE *adj_matrix) {

    std::srand(std::time(nullptr));

    for (int r = 0; r < NNODES; ++r)
        for (int c = 0; c < NNODES; ++c) {
            if (c == r) {
                adj_matrix[r*NNODES + c] = 0;
                continue;
            }
            adj_matrix[r*NNODES + c] = static_cast <ADJ_DTYPE> (roundf(
                static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX)
            ));
        }

}

// initialize feature matrix
void feature_matrix_init(FEATURE_DTYPE *feature_matrix) {

    std::srand(std::time(nullptr));

    for (int n = 0; n < NNODES; ++n)
        for (int d = 0; d < FEATURE_DIM; ++d)
            feature_matrix[n*FEATURE_DIM + d] = 
                static_cast <FEATURE_DTYPE> (std::rand()) / static_cast <FEATURE_DTYPE> (RAND_MAX) - 0.5f;

}

// print adjacency matrix
void print_adj_matrix(ADJ_DTYPE *adj_matrix) {

    std::cout<<"ADJ MATRIX:"<<std::endl;
    for (int r = 0; r < NNODES; ++r) {
        for (int c = 0; c < NNODES; ++c)
            std::cout<<adj_matrix[r*NNODES + c]<<" ";
        std::cout<<std::endl;
    }

}

// print feature matrix
void print_feature_matrix(FEATURE_DTYPE *feature_matrix) {

    std::cout<<"FEATURES:"<<std::endl;
    for (int n = 0; n < NNODES; ++n) {
        for (int d = 0; d < FEATURE_DIM; ++d)
            std::cout<<feature_matrix[n*FEATURE_DIM + d]<<" ";
        std::cout<<std::endl;
    }

}

int main(void) {

    // set display precision
    std::cout<<std::fixed;
    std::cout<<std::setprecision(4);

    // pointer to the adjacency matrix
    ADJ_DTYPE *adj_matrix;
    // pointer to feature vector matrix
    FEATURE_DTYPE *feature_matrix;

    // size of the adjacency matrix
    size_t size_adj = 1L*NNODES*NNODES;
    // size of feature matrix
    size_t size_fm = 1L*NNODES*FEATURE_DIM;

    //allocate memory for the adjacency matrix
    adj_matrix = static_cast <ADJ_DTYPE *> (malloc(size_adj * sizeof(ADJ_DTYPE)));
    MALLOC_ERR_CHECK(adj_matrix);

    // allocate memory for feature matrix
    feature_matrix = static_cast <FEATURE_DTYPE *> (malloc(size_fm * sizeof(FEATURE_DTYPE)));
    MALLOC_ERR_CHECK(feature_matrix);

    // init adjacency matrix
    adj_matrix_init(adj_matrix);
    // init feature matrix
    feature_matrix_init(feature_matrix);

    // show adjacency matrix
    print_adj_matrix(adj_matrix);

    // obtain CSR representation of adjacency matrix
    CSR csr(adj_matrix);
    csr.dump();

    // obtain CSC representation of adjacency matrix
    CSC csc(adj_matrix);
    csc.dump();

    // show feature matrix
    print_feature_matrix(feature_matrix);

    // free adjacency matrix on exit
    free(adj_matrix);
    // free feature matrix
    free(feature_matrix);


    return 0;
}
