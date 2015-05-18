// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_ISTLEXCEPTION_HH
#define DUNE_ISTL_ISTLEXCEPTION_HH

#include <dune/common/exceptions.hh>

namespace Dune {

  /**
              @addtogroup ISTL
              @{
   */

  //! derive error class from the base class in common
  class ISTLError : public Dune::MathError {};

  //! Error specific to BCRSMatrix.
  class BCRSMatrixError
    : public ISTLError
  {};

  //! The overflow error used during implicit BCRSMatrix construction was exhausted.
  /**
   * This error occurs if the overflow area of the BCRSMatrix
   * did not have room for another non-zero entry during implicit
   * mode construction.
   *
   * You can fix this problem by either increasing the average row size
   * or the overflow fraction.
   */
  class ImplicitModeOverflowExhausted
    : public BCRSMatrixError
  {};

  /** @} end documentation */

} // end namespace

#endif
