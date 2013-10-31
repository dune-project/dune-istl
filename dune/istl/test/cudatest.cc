// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <iostream>
#include <cstdlib>
#include <dune/common/memory/cuda_allocator.hh>

using namespace Dune;

template <typename DT_>
int backend_test()
{
  int result(EXIT_SUCCESS);

  size_t size(4711);
  Memory::CudaAllocator<DT_> alloc;
  DT_ * device(alloc.allocate(size));
  DT_ * host(new DT_[size]);
  for (size_t i(0) ; i < size ; ++i)
    host[i] = i;

  Cuda::upload(device, host, size);
  for (size_t i(0) ; i < size ; ++i)
    host[i] = -1;
  Cuda::download(host, device, size);

  for (size_t i(0) ; i < size ; ++i)
  {
    if (host[i] != i)
      result = EXIT_FAILURE;
  }

  delete[] host;
  alloc.deallocate(device);
  return result;
}


int main()
{
  if (backend_test<float>() == EXIT_FAILURE)
  {
    std::cout<<"backend test failed!"<<std::endl;
    return EXIT_FAILURE;
  }
  if (backend_test<double>() == EXIT_FAILURE)
  {
    std::cout<<"backend test failed!"<<std::endl;
    return EXIT_FAILURE;
  }
  if (backend_test<size_t>() == EXIT_FAILURE)
  {
    std::cout<<"backend test failed!"<<std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
