// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_COMMON_MEMORY_CUDA_ALLOCATOR_HH
#define DUNE_COMMON_MEMORY_CUDA_ALLOCATOR_HH

#include <memory>

#include <dune/common/memory/domain.hh>

namespace Dune {
  namespace Memory {

    template <typename DT_> class CudaAllocator : public std::allocator<DT_>
    {
      public:
        template <class Type> struct rebind
        {
          typedef CudaAllocator<Type> other;
        };

        typename std::allocator<DT_>::pointer allocate (size_t n, std::allocator<void>::const_pointer hint=0);
        void deallocate (typename std::allocator<DT_>::pointer p, size_t n = 0);

    };

    template<typename T>
      struct allocator_domain<
      CudaAllocator<T>
      >
      {
        typedef Domain::CUDA type;
      };


  } // namespace Memory
} //namespace Dune

#endif // DUNE_COMMON_MEMORY_BLOCKED_ALLOCATOR_HH
