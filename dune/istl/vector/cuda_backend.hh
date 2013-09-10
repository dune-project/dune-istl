// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_VECTOR_CUDA_BACKEND_HH
#define DUNE_ISTL_VECTOR_CUDA_BACKEND_HH

#include <memory>

namespace Dune
{
  namespace Cuda
  {
    template <typename DT_> class CudaAllocator : public std::allocator<DT_>
    {
      public:
      typename std::allocator<DT_>::pointer allocate (size_t n, std::allocator<void>::const_pointer hint=0);
      void deallocate (typename std::allocator<DT_>::pointer p, size_t n = 0);
    };

    template <typename DT_>
    void upload(DT_ * dst, const DT_ * src, size_t count);

    template <typename DT_>
    void download(DT_ * dst, const DT_ * src, size_t count);
  }
}

#endif
