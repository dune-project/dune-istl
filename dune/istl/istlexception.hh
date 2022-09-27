// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_ISTLEXCEPTION_HH
#define DUNE_ISTL_ISTLEXCEPTION_HH

#include <dune/common/exceptions.hh>
#include <dune/common/fmatrix.hh>

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

  /** \brief Thrown when the compression buffer used by the implicit BCRSMatrix construction is exhausted
   *
   * This error occurs if the compression buffer of the BCRSMatrix
   * did not have room for another non-zero entry during implicit
   * mode construction.
   *
   * You can fix this problem by either increasing the average row size
   * or the compressionBufferSize value.
   */
  class ImplicitModeCompressionBufferExhausted
    : public BCRSMatrixError
  {};

  //! Thrown when a solver aborts due to some problem.
  /**
   * Problems that may cause the solver to abort include a NaN detected during
   * the convergence check (which may be caused by invalid input data), or
   * breakdown conditions (which can happen e.g. in BiCGSTABSolver or
   * RestartedGMResSolver).
   */
  class SolverAbort : public ISTLError {};

  //! Error when performing an operation on a matrix block
  /**
   * For example an error in a block LU decomposition
   */
  class MatrixBlockError : public virtual Dune::FMatrixError {
  public:
    int r, c; // row and column index of the entry from which the error resulted
  };

  /** @} end documentation */

} // end namespace

#endif
