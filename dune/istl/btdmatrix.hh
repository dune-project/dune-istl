// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BTDMATRIX_HH
#define DUNE_ISTL_BTDMATRIX_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/scalarvectorview.hh>
#include <dune/common/scalarmatrixview.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/blocklevel.hh>

/** \file
    \author Oliver Sander
    \brief Implementation of the BTDMatrix class
 */

namespace Dune {
  /**
   * @addtogroup ISTL_SPMV
   * @{
   */
  /** \brief A block-tridiagonal matrix

     \todo It would be safer and more efficient to have a real implementation of
     a block-tridiagonal matrix and not just subclassing from BCRSMatrix.  But that's
     quite a lot of work for that little advantage.*/
  template <class B, class A=std::allocator<B> >
  class BTDMatrix : public BCRSMatrix<B,A>
  {
  public:

    //===== type definitions and constants

    //! export the type representing the field
    using field_type = typename Imp::BlockTraits<B>::field_type;

    //! export the type representing the components
    typedef B block_type;

    //! export the allocator type
    typedef A allocator_type;

    //! implement row_type with compressed vector
    //typedef BCRSMatrix<B,A>::row_type row_type;

    //! The type for the index access and the size
    typedef typename A::size_type size_type;

    //! increment block level counter
    [[deprecated("Use free blockLevel function. Will be removed after 2.8.")]]
    static constexpr auto blocklevel = blockLevel<B>()+1;

    /** \brief Default constructor */
    BTDMatrix() : BCRSMatrix<B,A>() {}

    explicit BTDMatrix(size_type size)
      : BCRSMatrix<B,A>(size, size, BCRSMatrix<B,A>::random)
    {
      // Set number of entries for each row
      // All rows get three entries, except for the first and the last one
      for (size_t i=0; i<size; i++)
        this->BCRSMatrix<B,A>::setrowsize(i, 3 - (i==0) - (i==(size-1)));

      this->BCRSMatrix<B,A>::endrowsizes();

      // The actual entries for each row
      for (size_t i=0; i<size; i++) {
        if (i>0)
          this->BCRSMatrix<B,A>::addindex(i, i-1);
        this->BCRSMatrix<B,A>::addindex(i, i  );
        if (i<size-1)
          this->BCRSMatrix<B,A>::addindex(i, i+1);
      }

      this->BCRSMatrix<B,A>::endindices();
    }

    /** \brief Resize the matrix.  Invalidates the content! */
    void setSize(size_type size)
    {
      auto nonZeros = (size==0) ? 0 : size + 2*(size-1);
      this->BCRSMatrix<B,A>::setSize(size,   // rows
                                     size,   // columns
                                     nonZeros);

      // Set number of entries for each row
      // All rows get three entries, except for the first and the last one
      for (size_t i=0; i<size; i++)
        this->BCRSMatrix<B,A>::setrowsize(i, 3 - (i==0) - (i==(size-1)));

      this->BCRSMatrix<B,A>::endrowsizes();

      // The actual entries for each row
      for (size_t i=0; i<size; i++) {
        if (i>0)
          this->BCRSMatrix<B,A>::addindex(i, i-1);
        this->BCRSMatrix<B,A>::addindex(i, i  );
        if (i<size-1)
          this->BCRSMatrix<B,A>::addindex(i, i+1);
      }

      this->BCRSMatrix<B,A>::endindices();
    }

    //! assignment
    BTDMatrix& operator= (const BTDMatrix& other) {
      this->BCRSMatrix<B,A>::operator=(other);
      return *this;
    }

    //! assignment from scalar
    BTDMatrix& operator= (const field_type& k) {
      this->BCRSMatrix<B,A>::operator=(k);
      return *this;
    }

    /** \brief Use the Thomas algorithm to solve the system Ax=b in O(n) time
     *
     * \exception ISTLError if the matrix is singular
     *
     */
    template <class V>
    void solve (V& x, const V& rhs) const {

      // special handling for 1x1 matrices.  The generic algorithm doesn't work for them
      if (this->N()==1) {
        auto&& x0 = Impl::asVector(x[0]);
        auto&& rhs0 = Impl::asVector(rhs[0]);
        Impl::asMatrix((*this)[0][0]).solve(x0, rhs0);
        return;
      }

      // Make copies of the rhs and the right matrix band
      V d = rhs;
      std::vector<block_type> c(this->N()-1);
      for (size_t i=0; i<this->N()-1; i++)
        c[i] = (*this)[i][i+1];

      /* Modify the coefficients. */
      block_type a_00_inv = (*this)[0][0];
      Impl::asMatrix(a_00_inv).invert();

      //c[0] /= (*this)[0][0]; /* Division by zero risk. */
      block_type tmp = a_00_inv;
      Impl::asMatrix(tmp).rightmultiply(Impl::asMatrix(c[0]));
      c[0] = tmp;

      // d = a^{-1} d        /* Division by zero would imply a singular matrix. */
      auto d_0_tmp = d[0];
      auto&& d_0 = Impl::asVector(d[0]);
      Impl::asMatrix(a_00_inv).mv(Impl::asVector(d_0_tmp),d_0);

      for (unsigned int i = 1; i < this->N(); i++) {

        // id = ( a_ii - c_{i-1} a_{i, i-1} ) ^{-1}
        block_type tmp;
        tmp = (*this)[i][i-1];
        Impl::asMatrix(tmp).rightmultiply(Impl::asMatrix(c[i-1]));

        block_type id = (*this)[i][i];
        id -= tmp;
        Impl::asMatrix(id).invert(); /* Division by zero risk. */

        if (i<c.size()) {
          Impl::asMatrix(c[i]).leftmultiply(Impl::asMatrix(id));            /* Last value calculated is redundant. */
        }

        // d[i] = (d[i] - d[i-1] * (*this)[i][i-1]) * id;
        auto&& d_i = Impl::asVector(d[i]);
        Impl::asMatrix((*this)[i][i-1]).mmv(Impl::asVector(d[i-1]), d_i);
        auto tmpVec = d[i];
        Impl::asMatrix(id).mv(Impl::asVector(tmpVec), d_i);
      }

      /* Now back substitute. */
      x[this->N() - 1] = d[this->N() - 1];
      for (int i = this->N() - 2; i >= 0; i--) {
        //x[i] = d[i] - c[i] * x[i + 1];
        x[i] = d[i];
        auto&& x_i = Impl::asVector(x[i]);
        Impl::asMatrix(c[i]).mmv(Impl::asVector(x[i+1]), x_i);
      }

    }

  private:

    // ////////////////////////////////////////////////////////////////////////////
    //   The following methods from the base class should now actually be called
    // ////////////////////////////////////////////////////////////////////////////

    // createbegin and createend should be in there, too, but I can't get it to compile
    //     BCRSMatrix<B,A>::CreateIterator createbegin () {}
    //     BCRSMatrix<B,A>::CreateIterator createend () {}
    void setrowsize (size_type i, size_type s) {}
    void incrementrowsize (size_type i) {}
    void endrowsizes () {}
    void addindex (size_type row, size_type col) {}
    void endindices () {}
  };

  template<typename B, typename A>
  struct FieldTraits< BTDMatrix<B, A> >
  {
    using field_type = typename BTDMatrix<B, A>::field_type;
    using real_type = typename FieldTraits<field_type>::real_type;
  };

  /** @}*/

}  // end namespace Dune

#endif
