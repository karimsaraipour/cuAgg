#ifndef SRC__CUDA_CUH
#define SRC__CUDA_CUH

#include <cstdlib>
#include <iostream>

inline static void CUErrCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define CUDA_ERRCHK(err) (CUErrCheck(err, __FILE__, __LINE__))

#endif // SRC__CUDA_CUH
