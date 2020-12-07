#include <vector>
#include <iostream>
#include <string>

#include "memkind.h"
#include "pmem_allocator.h"

using namespace libmemkind;

int main() {
  auto allocator = pmem::allocator<int>(".", MEMKIND_PMEM_MIN_SIZE);
  std::vector<int, pmem::allocator<int>> v(allocator);
  for (size_t i = 0; i < 3; ++i)
    v.push_back(42);

  for (auto value : v)
    std::cout << value << std::endl;
  return 0;
}
