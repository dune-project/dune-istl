// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_COLCOMPMATRIX_HH
#define DUNE_ISTL_COLCOMPMATRIX_HH

#warning "Deprecated header, include <dune/istl/bccsmatrixinitializer.hh> instead!"

#include <dune/istl/bccsmatrixinitializer.hh>

namespace Dune
{
  /** \brief A sparse matrix in compressed-column format
   *
   * \deprecated This class has been superseded by Impl::BCCSMatrix
   *   and Impl::BCCSMatrixInitializer.  Please use that instead!
   */
  template <class M, class I = int>
  struct
  [[deprecated]] ColCompMatrix : public ISTL::Impl::BCCSMatrix< typename M::field_type, I >
  {};
}
#endif
