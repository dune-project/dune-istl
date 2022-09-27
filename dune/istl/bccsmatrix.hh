// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BCCSMATRIX_HH
#define DUNE_ISTL_BCCSMATRIX_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/typetraits.hh>

namespace Dune::ISTL::Impl
{
  /**
   * @brief A block matrix with compressed-column storage
   *
   * @tparam B The type of a matrix entry (may be a matrix itself)
   * @tparam I The type used for row and column indices.
   *   When using BCCSMatrix to set up a SuiteSparse solver, this type
   *   should be set to the index type used by the solver.  That way,
   *   the solver can use the row/column information without any copying.
   *
   * Currently this class is mainly a vehicle to get matrices into the solvers
   * of the SuiteSparse package.  The class tries to follow the dune-istl
   * matrix interface, but it doesn't implement the interface entirely.
   */
  template<class B, class I = typename std::allocator<B>::size_type>
  class BCCSMatrix
  {
  public:
    using Index = I;
    using size_type = std::size_t;

    /** \brief Default constructor
     */
    BCCSMatrix()
    : N_(0), M_(0), Nnz_(0), values(0), rowindex(0), colstart(0)
    {}

    /** @brief Destructor */
    ~BCCSMatrix()
    {
      if(N_+M_+Nnz_!=0)
        free();
    }

    /** \brief Set matrix size */
    void setSize(size_type rows, size_type columns)
    {
      N_ = rows;
      M_ = columns;
    }

    /**
     * @brief Get the number of rows.
     * @return  The number of rows.
     */
    size_type N() const
    {
      return N_;
    }

    /** \brief Return the number of nonzero block entries */
    size_type nonzeroes() const
    {
      return Nnz_;
    }

    /**
     * @brief Get the number of columns.
     * @return  The number of columns.
     */
    size_type M() const
    {
      return M_;
    }

    /** \brief Direct access to the array of matrix entries
     *
     * This is meant to be handed directly to SuiteSparse solvers.
     * Note that the 'Index' type is controllable via the second
     * class template argument.
     */
    B* getValues() const
    {
      return values;
    }

    /** \brief Direct access to the array of row indices
     *
     * This is meant to be handed directly to SuiteSparse solvers.
     * Note that the 'Index' type is controllable via the second
     * class template argument.
     */
    Index* getRowIndex() const
    {
      return rowindex;
    }

    /** \brief Direct access to the array of column entry points
     *
     * This is meant to be handed directly to SuiteSparse solvers.
     * Note that the 'Index' type is controllable via the second
     * class template argument.
     */
    Index* getColStart() const
    {
      return colstart;
    }

    /** \brief Assignment operator */
    BCCSMatrix& operator=(const BCCSMatrix& mat)
    {
      if(N_+M_+Nnz_!=0)
        free();
      N_=mat.N_;
      M_=mat.M_;
      Nnz_= mat.Nnz_;
      if(M_>0) {
        colstart=new size_type[M_+1];
        for(size_type i=0; i<=M_; ++i)
          colstart[i]=mat.colstart[i];
      }

      if(Nnz_>0) {
        values = new B[Nnz_];
        rowindex = new size_type[Nnz_];

        for(size_type i=0; i<Nnz_; ++i)
          values[i]=mat.values[i];

        for(size_type i=0; i<Nnz_; ++i)
          rowindex[i]=mat.rowindex[i];
      }
      return *this;
    }

    /** @brief free allocated space. */
    virtual void free()
    {
      delete[] values;
      delete[] rowindex;
      delete[] colstart;
      N_ = 0;
      M_ = 0;
      Nnz_ = 0;
    }

  public:
    size_type N_, M_, Nnz_;
    B* values;
    Index* rowindex;
    Index* colstart;
  };

}
#endif
