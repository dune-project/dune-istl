// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_DILU_HH
#define DUNE_ISTL_DILU_HH

#include <cmath>
#include <complex>
#include <map>
#include <vector>
#include <sstream>

#include <dune/common/fmatrix.hh>
#include <dune/common/scalarvectorview.hh>
#include <dune/common/scalarmatrixview.hh>

#include "istlexception.hh"

/** \file
 * \brief  The diagonal incomplete LU factorization kernels
 */

namespace Dune
{

  /** @addtogroup ISTL_Kernel
          @{
   */

  namespace DILU
  {

    /*! compute DILU decomposition of A

        The preconditioner matrix M has the property

        diag(A) = diag(M) = diag((D + L_A) D^{-1} (D + U_A)) = diag(D + L_A D^{-1} U_A)

        such that the diagonal matrix D can be constructed:

        D_11 = A_11
        D_22 = A22 - L_A_{21} D_{11}^{-1} U_A_{12}
        and etc

        we store the inverse of D to be used when applying the preconditioner.

        For more details, see: R. Barrett et al., "Templates for the Solution of Linear Systems:
        Building Blocks for Iterative Methods", 1994. Available from: https://www.netlib.org/templates/templates.pdf
     */
    template <class M>
    void blockDILUDecomposition(M &A, std::vector<typename M::block_type> &Dinv_)
    {
      auto endi = A.end();
      for (auto row = A.begin(); row != endi; ++row)
      {
        const auto row_i = row.index();
        const auto col = row->find(row_i);
        // initialise Dinv[i] = A[i, i]
        Dinv_[row_i] = *col;
      }

      for (auto row = A.begin(); row != endi; ++row)
      {
        const auto row_i = row.index();
        for (auto a_ij = row->begin(); a_ij.index() < row_i; ++a_ij)
        {
          const auto col_j = a_ij.index();
          const auto a_ji = A[col_j].find(row_i);
          // if A[i, j] != 0 and A[j, i] != 0
          if (a_ji != A[col_j].end())
          {
            // Dinv[i] -= A[i, j] * d[j] * A[j, i]
            Dinv_[row_i] -= (*a_ij) * Dinv_[col_j] * (*a_ji);
          }
        }

        // store the inverse
        try
        {
          Impl::asMatrix(Dinv_[row_i]).invert(); // compute inverse of diagonal block
        }
        catch (Dune::FMatrixError &e)
        {
          std::ostringstream sstream;
          sstream << THROWSPEC(MatrixBlockError)
            << "DILU failed to invert matrix block D[" << row_i << "]" << e.what();
          MatrixBlockError ex;
          ex.message(sstream.str());
          ex.r = row_i;
          throw ex;
        }
      }
    }

    /*! DILU backsolve

      M = (D + L_A) D^-1 (D + U_A)   (a LU decomposition of M)
      where L_A and U_A are the strictly lower and upper parts of A.

      Working with residual d = b - Ax and update v = x_{n+1} - x_n
      solving the product M^-1(d) using upper and lower triangular solve:

      v = M^{-1} d = (D + U_A)^{-1} D (D + L_A)^{-1}  d

      define y = (D + L_A)^{-1} d

      lower triangular solve: (D + L_A) y = d
      upper triangular solve: (D + U_A) v = D y
     */
    template <class M, class X, class Y>
    void blockDILUBacksolve(const M &A, const std::vector<typename M::block_type> Dinv_, X &v, const Y &d)
    {
      using dblock = typename Y::block_type;
      using vblock = typename X::block_type;

      // lower triangular solve: (D + L_A) y = d
      auto endi = A.end();
      for (auto row = A.begin(); row != endi; ++row)
      {
        const auto row_i = row.index();
        dblock rhsValue(d[row_i]);
        auto &&rhs = Impl::asVector(rhsValue);
        for (auto a_ij = (*row).begin(); a_ij.index() < row_i; ++a_ij)
        {
          // if  A[i][j] != 0
          // rhs -= A[i][j]* y[j], where v_j stores y_j
          const auto col_j = a_ij.index();
          Impl::asMatrix(*a_ij).mmv(v[col_j], rhs);
        }
        // y_i = Dinv_i * rhs
        // storing y_i in v_i
        auto &&vi = Impl::asVector(v[row_i]);
        Impl::asMatrix(Dinv_[row_i]).mv(rhs, vi); // (D + L_A)_ii = D_i
      }

      // upper triangular solve: (D + U_A) v = Dy
      auto rendi = A.beforeBegin();
      for (auto row = A.beforeEnd(); row != rendi; --row)
      {
        const auto row_i = row.index();
        // rhs = 0
        vblock rhs(0.0);
        for (auto a_ij = (*row).beforeEnd(); a_ij.index() > row_i; --a_ij)
        {
          // if A[i][j] != 0
          // rhs += A[i][j]*v[j]
          const auto col_j = a_ij.index();
          Impl::asMatrix(*a_ij).umv(v[col_j], rhs);
        }
        // calculate update v = M^-1*d
        // v_i = y_i - Dinv_i*rhs
        // before update v_i is y_i
        auto &&vi = Impl::asVector(v[row_i]);
        Impl::asMatrix(Dinv_[row_i]).mmv(rhs, vi);
      }
    }
  } // end namespace DILU

  /** @} end documentation */

} // end namespace

#endif
