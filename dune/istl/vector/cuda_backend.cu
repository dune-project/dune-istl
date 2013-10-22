#include <cuda.h>
#include <dune/istl/vector/cuda_backend.hh>

using namespace Dune;
using namespace Dune::Cuda;

template <typename DT_>
void Dune::Cuda::upload(DT_ * dst, const DT_ * src, size_t count)
{
  cudaMemcpy(dst, src, count * sizeof(DT_), cudaMemcpyHostToDevice);
}

template <typename DT_>
void Dune::Cuda::download(DT_ * dst, const DT_ * src, size_t count)
{
  cudaMemcpy(dst, src, count * sizeof(DT_), cudaMemcpyDeviceToHost);
}

template <typename DT_>
void Dune::Cuda::copy(DT_ * dst, const DT_ * src, size_t count)
{
  cudaMemcpy(dst, src, count * sizeof(DT_), cudaMemcpyDeviceToDevice);
}

template <typename DT_>
void Dune::Cuda::set(DT_ * dst, const DT_ & val)
{
  cudaMemcpy(dst, &val, sizeof(DT_), cudaMemcpyHostToDevice);
}

template <typename DT_>
DT_ Dune::Cuda::get(DT_ * src)
{
  DT_ result;
  cudaMemcpy(&result, src, sizeof(DT_), cudaMemcpyDeviceToHost);
  return result;
}

template void Dune::Cuda::upload(float *, const float *, size_t);
template void Dune::Cuda::upload(double *, const double *, size_t);
template void Dune::Cuda::download(float *, const float *, size_t);
template void Dune::Cuda::download(double *, const double *, size_t);
template void Dune::Cuda::copy(float *, const float *, size_t);
template void Dune::Cuda::copy(double *, const double *, size_t);
template void Dune::Cuda::set(float *, const float &);
template void Dune::Cuda::set(double *, const double &);
template float Dune::Cuda::get(float *);
template double Dune::Cuda::get(double *);
