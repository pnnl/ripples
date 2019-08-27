#include <iostream>
#include "trng/lcg64.hpp"

int main() {
  trng::lcg64 generator;
  generator.seed(0UL);
  generator.split(2, 1);
  return 0;
}
