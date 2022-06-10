#include "testing.h"

#include <iostream>
#include <math.h>

bool check(size_t i, float test, float oracle) {
  bool is_correct = fabs(test - oracle) < 0.005;
  if (!is_correct) {
    std::cerr << "Test failed at index " << i << std::endl;
    std::cerr << "CPU: " << test << std::endl;
    std::cerr << "GPU: " << oracle << std::endl;
    std::cerr << "Delta: " << fabs(test - oracle) << std::endl;
  }
  return is_correct;
}
