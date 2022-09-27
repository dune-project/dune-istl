// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BDMATRIX_HH
#define DUNE_ISTL_BDMATRIX_HH

#include <memory>

#include <dune/common/rangeutilities.hh>
#include <dune/common/scalarmatrixview.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/blocklevel.hh>

/** \file
    \author Oliver Sander
    \brief Implementation of the BDMatrix class
 */

namespace Dune {
  /**
   * @addtogroup ISTL_SPMV
   * @{
   */
  /** \brief A block-diagonal matrix

     \todo It would be safer and more efficient to have a real implementation of
     a block-diagonal matrix and not just subclassing from BCRSMatrix.  But that's
     quite a lot of work for that little advantage.*/
  template <class B, class A=std::allocator<B> >
  class BDMatrix : public BCRSMatrix<B,A>
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
    [[deprecated("Use free function blockLevel(). Will be removed after 2.8.")]]
    static constexpr unsigned int blocklevel = blockLevel<B>()+1;

    /** \brief Default constructor */
    BDMatrix() : BCRSMatrix<B,A>() {}

    explicit BDMatrix(int size)
      : BCRSMatrix<B,A>(size, size, BCRSMatrix<B,A>::random) {

      for (int i=0; i<size; i++)
        this->BCRSMatrix<B,A>::setrowsize(i, 1);

      this->BCRSMatrix<B,A>::endrowsizes();

      for (int i=0; i<size; i++)
        this->BCRSMatrix<B,A>::addindex(i, i);

      this->BCRSMatrix<B,A>::endindices();

    }

    /** \brief Construct from a std::initializer_list */
    BDMatrix (std::initializer_list<B> const &list)
    : BDMatrix(list.size())
    {
      size_t i=0;
      for (auto it = list.begin(); it != list.end(); ++it, ++i)
        (*this)[i][i] = *it;
    }

    /** \brief Resize the matrix.  Invalidates the content! */
    void setSize(size_type size)
    {
      this->BCRSMatrix<B,A>::setSize(size,   // rows
                                     size,   // columns
                                     size);  // nonzeros

      for (auto i : range(size))
        this->BCRSMatrix<B,A>::setrowsize(i, 1);

      this->BCRSMatrix<B,A>::endrowsizes();

      for (auto i : range(size))
        this->BCRSMatrix<B,A>::addindex(i, i);

      this->BCRSMatrix<B,A>::endindices();
    }

    //! assignment
    BDMatrix& operator= (const BDMatrix& other) {
      this->BCRSMatrix<B,A>::operator=(other);
      return *this;
    }

    //! assignment from scalar
    BDMatrix& operator= (const field_type& k) {
      this->BCRSMatrix<B,A>::operator=(k);
      return *this;
    }

    /** \brief Solve the system Ax=b in O(n) time
     *
     * \exception ISTLError if the matrix is singular
     *
     */
    template <class V>
    void solve (V& x, const V& rhs) const {
      for (size_type i=0; i<this->N(); i++)
      {
        auto&& xv = Impl::asVector(x[i]);
        auto&& rhsv = Impl::asVector(rhs[i]);
        Impl::asMatrix((*this)[i][i]).solve(xv,rhsv);
      }
    }

    /** \brief Inverts the matrix */
    void invert() {
      for (size_type i=0; i<this->N(); i++)
        Impl::asMatrix((*this)[i][i]).invert();
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
  struct FieldTraits< BDMatrix<B, A> >
  {
    using field_type = typename BDMatrix<B, A>::field_type;
    using real_type = typename FieldTraits<field_type>::real_type;
  };
  /** @}*/

}  // end namespace Dune

#endif
