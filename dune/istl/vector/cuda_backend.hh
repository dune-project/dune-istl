// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_VECTOR_CUDA_BACKEND_HH
#define DUNE_ISTL_VECTOR_CUDA_BACKEND_HH

#include <memory>

namespace Dune
{
  namespace Cuda
  {
    template <typename DT_>
    void upload(DT_ * dst, const DT_ * src, size_t count);

    template <typename DT_>
    void download(DT_ * dst, const DT_ * src, size_t count);

    template <typename DT_>
    void copy(DT_ * dst, const DT_ * src, size_t count);

    template <typename DT_>
    void set(DT_ * dst, const DT_ & val);

    template <typename DT_>
    DT_ get(DT_ * src);
  }
}

#endif
