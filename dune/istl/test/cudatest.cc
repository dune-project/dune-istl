// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/vector/cuda_backend.hh>
#include <cstdlib>

using namespace Dune;

template <typename DT_>
int test()
{
  int result(EXIT_SUCCESS);

  unsigned long size(11);
  Cuda::CudaAllocator<DT_> alloc;
  DT_ * device(alloc.allocate(size));
  DT_ * host(new DT_[size]);
  for (unsigned long i(0) ; i < size ; ++i)
    host[i] = i;

  Cuda::upload(device, host, size);
  for (unsigned long i(0) ; i < size ; ++i)
    host[i] = -1;
  Cuda::download(host, device, size);

  for (unsigned long i(0) ; i < size ; ++i)
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
  if (test<float>() == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test<double>() == EXIT_FAILURE)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
