#include <cuda.h>
#include <dune/istl/vector/cuda_allocator.hh>

using namespace Dune;
using namespace Dune::Memory;

template <typename DT_>
typename std::allocator<DT_>::pointer CudaAllocator<DT_>::allocate(size_t n, typename std::allocator<void>::const_pointer /*hint*/)
{
  void * r;
  cudaError_t status = cudaMalloc(&r, n * sizeof(DT_));
  if (status != cudaSuccess)
    throw new std::bad_alloc;

  return (DT_*)r;
}

template <typename DT_>
void CudaAllocator<DT_>::deallocate(typename std::allocator<DT_>::pointer p, size_t /*n*/)
{
  cudaFree((void*) p);
}

template typename std::allocator<float>::pointer CudaAllocator<float>::allocate(size_t n, typename std::allocator<void>::const_pointer);
template typename std::allocator<double>::pointer CudaAllocator<double>::allocate(size_t n, typename std::allocator<void>::const_pointer);
template void CudaAllocator<float>::deallocate(typename std::allocator<float>::pointer, size_t);
template void CudaAllocator<double>::deallocate(typename std::allocator<double>::pointer, size_t);
