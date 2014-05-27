// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <iostream>
#include <cstdlib>
#include <dune/istl/vector/cuda.hh>
#include <dune/common/memory/blocked_allocator.hh>
#include <dune/istl/vector/host.hh>

using namespace Dune;

template <typename DT_, typename A_>
int vector_test()
{
  int result(EXIT_SUCCESS);

  size_t size(4711);
  ISTL::Vector<DT_, A_> v1;
  if (v1.size() != 0)
    return EXIT_FAILURE;

  ISTL::Vector<DT_, A_> v2(size);
  for (size_t i(0) ; i < size ; ++i)
    v2(i, i);
  if (v2.size() != size)
    return EXIT_FAILURE;
  for (size_t i(0) ; i < size ; ++i)
    if (v2[i] != i)
      return EXIT_FAILURE;

  ISTL::Vector<DT_, A_> v3(v2);
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

  ISTL::Vector<DT_, A_> v4(size);
  ISTL::Vector<DT_, A_> v5(size);
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
    if (fabs(v4[i] - (v5[i] * DT_(1.23) + DT_(i+1))) > 1e-1)
      return EXIT_FAILURE;

  for (size_t i(0) ; i < size ; ++i)
  {
    v4(i, i+1);
    v5(i, DT_(1) / DT_(i+1));
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
  if (norm != 4711)
    return EXIT_FAILURE;

  return result;
}

template <typename DT_>
int vector_regression()
{
  int result(EXIT_SUCCESS);

  size_t size(4711);
  ISTL::Vector<DT_, Dune::Memory::blocked_cache_aligned_allocator<DT_,std::size_t,16> > v1_cpu(size, DT_(1.234));
  ISTL::Vector<DT_, Memory::CudaAllocator<DT_> > v1_gpu(v1_cpu);
  ISTL::Vector<DT_, Dune::Memory::blocked_cache_aligned_allocator<DT_,std::size_t,16> > v2_cpu(size, DT_(-2.234));
  ISTL::Vector<DT_, Memory::CudaAllocator<DT_> > v2_gpu(v2_cpu);

  ISTL::Vector<DT_, Dune::Memory::blocked_cache_aligned_allocator<DT_,std::size_t,16> > v1_1_cpu(size, DT_(4711));
  v1_gpu.download_to(v1_1_cpu);
  for (size_t i(0) ; i < size ; ++i)
  {
    if (fabs(v1_1_cpu[i] - v1_cpu[i]) > 1e7)
    {
      return EXIT_FAILURE;
    }
  }

  v2_cpu += v1_cpu;
  v2_gpu += v1_gpu;
  for (size_t i(0) ; i < size ; ++i)
  {
    if (fabs(v2_gpu[i] - v2_cpu[i]) > 1e-7)
      return EXIT_FAILURE;
  }

  v2_cpu -= v1_cpu;
  v2_gpu -= v1_gpu;
  for (size_t i(0) ; i < size ; ++i)
  {
    if (fabs(v2_gpu[i] - v2_cpu[i]) > 1e-7)
      return EXIT_FAILURE;
  }

  v2_cpu *= v1_cpu;
  v2_gpu *= v1_gpu;
  for (size_t i(0) ; i < size ; ++i)
  {
    if (fabs(v2_gpu[i] - v2_cpu[i]) > 1e-7)
      return EXIT_FAILURE;
  }

  v2_cpu.axpy(DT_(5.678), v1_cpu);
  v2_gpu.axpy(DT_(5.678), v1_gpu);
  for (size_t i(0) ; i < size ; ++i)
  {
    if (fabs(v2_gpu[i] - v2_cpu[i]) > 1e-7)
      return EXIT_FAILURE;
  }

  for (size_t i(0) ; i < size ; ++i)
  {
    v1_cpu[i] = DT_(i);
    v2_cpu[i] = DT_(1) / DT_(i+1);
  }
  v1_gpu = v1_cpu;
  v2_gpu = v2_cpu;

  DT_ dot_cpu = v1_cpu.dot(v2_cpu);
  DT_ dot_gpu = v1_gpu.dot(v2_gpu);
  if (fabs(dot_cpu - dot_gpu) > 1e-2)
    return EXIT_FAILURE;

  DT_ norm_cpu = v2_cpu.two_norm2();
  DT_ norm_gpu = v2_gpu.two_norm2();
  if (fabs(norm_cpu - norm_gpu) > 1e-2)
    return EXIT_FAILURE;

  norm_cpu = v2_cpu.two_norm();
  norm_gpu = v2_gpu.two_norm();
  if (fabs(norm_cpu - norm_gpu) > 1e-2)
    return EXIT_FAILURE;

  norm_cpu = v2_cpu.one_norm();
  norm_gpu = v2_gpu.one_norm();
  if (fabs(norm_cpu - norm_gpu) > 1e-2)
    return EXIT_FAILURE;

  norm_cpu = v2_cpu.infinity_norm();
  norm_gpu = v2_gpu.infinity_norm();
  if (fabs(norm_cpu - norm_gpu) > 1e-2)
    return EXIT_FAILURE;

  return result;
}

int main()
{
  if (vector_test<float, Memory::CudaAllocator<float> >() == EXIT_FAILURE)
  {
    std::cout<<"vector test failed!"<<std::endl;
    return EXIT_FAILURE;
  }
  if (vector_test<double, Memory::CudaAllocator<double> >() == EXIT_FAILURE)
  {
    std::cout<<"vector test failed!"<<std::endl;
    return EXIT_FAILURE;
  }
  if (vector_regression<float>() == EXIT_FAILURE)
  {
    std::cout<<"vector regression test failed!"<<std::endl;
    return EXIT_FAILURE;
  }
  if (vector_regression<double>() == EXIT_FAILURE)
  {
    std::cout<<"vector regression test failed!"<<std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
