// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <iostream>
#include <dune/istl/vector/cuda.hh>
#include <cstdlib>

using namespace Dune;

template <typename DT_>
int backend_test()
{
  int result(EXIT_SUCCESS);

  size_t size(4711);
  Cuda::CudaAllocator<DT_> alloc;
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

template <typename DT_>
int vector_test()
{
  int result(EXIT_SUCCESS);

  size_t size(4711);
  ISTL::Vector<DT_, Cuda::CudaAllocator<DT_> > v1;
  if (v1.size() != 0)
    return EXIT_FAILURE;

  ISTL::Vector<DT_, Cuda::CudaAllocator<DT_> > v2(size);
  for (size_t i(0) ; i < size ; ++i)
    v2(i, i);
  if (v2.size() != size)
    return EXIT_FAILURE;
  for (size_t i(0) ; i < size ; ++i)
    if (v2[i] != i)
      return EXIT_FAILURE;

  ISTL::Vector<DT_, Cuda::CudaAllocator<DT_> > v3(v2);
  if (v3.size() != v2.size())
    return EXIT_FAILURE;
  for (size_t i(0) ; i < size ; ++i)
    if (v2[i] != i)
      return EXIT_FAILURE;

  v3(5, 7);
  if (v2[5] != 5)
    return EXIT_FAILURE;
  v3 = v2;
  for (size_t i(0) ; i < size ; ++i)
    if (v2[i] != i)
      return EXIT_FAILURE;

  ISTL::Vector<DT_, Cuda::CudaAllocator<DT_> > v4(size);
  ISTL::Vector<DT_, Cuda::CudaAllocator<DT_> > v5(size);
  for (size_t i(0) ; i < size ; ++i)
  {
    v4(i, i+1);
    v5(i, 2*(i+1));
  }

  v5+=v4;
  for (size_t i(0) ; i < size ; ++i)
    if (v5[i] != 3*(i+1))
      return EXIT_FAILURE;
  v5-=v4;
  for (size_t i(0) ; i < size ; ++i)
    if (v5[i] != 2*(i+1))
      return EXIT_FAILURE;

  v5*=v4;
  for (size_t i(0) ; i < size ; ++i)
    if (v5[i] != 2*(i+1)*(i+1))
      return EXIT_FAILURE;

  v5/=v4;
  for (size_t i(0) ; i < size ; ++i)
    if (fabs(v5[i] - 2*(i+1)) > 1e-1)
      return EXIT_FAILURE;

  v4+=DT_(1.23);
  for (size_t i(0) ; i < size ; ++i)
    if (v4[i] != DT_(i+1) + DT_(1.23))
      return EXIT_FAILURE;

  v4-=DT_(1.23);
  for (size_t i(0) ; i < size ; ++i)
    if (fabs(v4[i] - DT_(i+1)) > 1e-1)
      return EXIT_FAILURE;

  v4*=DT_(1.23);
  for (size_t i(0) ; i < size ; ++i)
    if (fabs(v4[i] - DT_(i+1) * DT_(1.23)) > 1e-1)
      return EXIT_FAILURE;

  v4/=DT_(1.23);
  for (size_t i(0) ; i < size ; ++i)
    if (fabs(v4[i] - DT_(i+1)) > 1e-1)
      return EXIT_FAILURE;

  v4.axpy(DT_(1.23), v5);
  for (size_t i(0) ; i < size ; ++i)
    if (fabs(v4[i] - (DT_(i+1) * DT_(1.23) + v5[i])) > 1e-1)
      return EXIT_FAILURE;

  for (size_t i(0) ; i < size ; ++i)
  {
    v4(i, i);
    v5(i, DT_(1) / DT_(i));
  }
  DT_ dot = v4.dot(v5);
  if (fabs(dot - size) > 1e-1)
    return EXIT_FAILURE;

  for (size_t i(0) ; i < size ; ++i)
  {
    v4(i, DT_(1.234));
    v5(i, DT_(1));
  }
  DT_ norm = v4.two_norm2();
  if (fabs(norm - v4.dot(v4)) > 1e-1)
    return EXIT_FAILURE;

  norm = v4.two_norm();
  if (fabs(norm - std::sqrt(v4.dot(v4))) > 1e-1)
    return EXIT_FAILURE;

  norm = v4.one_norm();
  if (fabs(norm - v4.dot(v5)) > 1e-1)
    return EXIT_FAILURE;

  v4(42, 4711);
  norm = v4.infinity_norm();
  std::cout<<norm<<" "<<4711<<std::endl;
  if (norm != 42)
    return EXIT_FAILURE;

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
  if (vector_test<float>() == EXIT_FAILURE)
  {
    std::cout<<"vector test failed!"<<std::endl;
    return EXIT_FAILURE;
  }
  if (vector_test<double>() == EXIT_FAILURE)
  {
    std::cout<<"vector test failed!"<<std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
