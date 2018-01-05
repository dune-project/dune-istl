// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BLOCKLEVEL_HH
#define DUNE_ISTL_BLOCKLEVEL_HH

#include <dune/common/typetraits.hh>

namespace Dune {

namespace Imp {

/** \brief Computes the nesting level of an ISTL matrix
 *
 * This assumes that there is such thing as a well-defined nesting level.
 * There is for BCRSMatrix and friends, but the nesting level of a
 * MultiTypeBlockMatrix depends on which entry is considered.
 */
template <class MatrixType>
constexpr int matrixBlockLevel()
{
  if constexpr (IsNumber<MatrixType>::value)
    return 0;
  else
    return matrixBlockLevel<typename MatrixType::block_type>() + 1;
}

}  // Namespace Imp

}  // Namespace Dune

#endif
