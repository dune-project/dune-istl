// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SCALEDIDMATRIX_HH
#define DUNE_ISTL_SCALEDIDMATRIX_HH

/*! \file

   \brief  This file implements a quadratic matrix of fixed size which is
   a multiple of the identity.
 */

#include <cmath>
#include <cstddef>
#include <complex>
#include <iostream>
#include <dune/common/exceptions.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/diagonalmatrix.hh>
#include <dune/common/ftraits.hh>

namespace Dune {

  /**
      @brief A multiple of the identity matrix of static size
   */
  template<class K, int n>
  class ScaledIdentityMatrix
  {
    typedef DiagonalMatrixWrapper< ScaledIdentityMatrix<K,n> > WrapperType;

  public:
    //===== type definitions and constants

    //! export the type representing the field
    typedef K field_type;

    //! export the type representing the components
    typedef K block_type;

    //! The type used for the index access and size operations.
    typedef std::size_t size_type;

    //! We are at the leaf of the block recursion
    [[deprecated("Use free function blockLevel(). Will be removed after 2.8.")]]
    static constexpr std::size_t blocklevel = 1;

    //! Each row is implemented by a field vector
    typedef DiagonalRowVector<K,n> row_type;
    typedef row_type reference;
    typedef DiagonalRowVectorConst<K,n> const_row_type;
    typedef const_row_type const_reference;

    //! export size
    enum {
      //! The number of rows.
      rows = n,
      //! The number of columns.
      cols = n
    };

    //===== constructors
    /** \brief Default constructor
     */
    ScaledIdentityMatrix () {}

    /** \brief Constructor initializing the whole matrix with a scalar
     */
    ScaledIdentityMatrix (const K& k)
      : p_(k)
    {}

    //===== assignment from scalar
    ScaledIdentityMatrix& operator= (const K& k)
    {
      p_ = k;
      return *this;
    }

    // check if matrix is identical to other matrix (not only identical values)
    bool identical(const ScaledIdentityMatrix<K,n>& other) const
    {
      return (this==&other);
    }

    //===== iterator interface to rows of the matrix
    //! Iterator class for sequential access
    typedef ContainerWrapperIterator<const WrapperType, reference, reference> Iterator;
    //! typedef for stl compliant access
    typedef Iterator iterator;
    //! rename the iterators for easier access
    typedef Iterator RowIterator;
    //! rename the iterators for easier access
    typedef typename row_type::Iterator ColIterator;

    //! begin iterator
    Iterator begin ()
    {
      return Iterator(WrapperType(this),0);
    }

    //! end iterator
    Iterator end ()
    {
      return Iterator(WrapperType(this),n);
    }

    //! @returns an iterator that is positioned before
    //! the end iterator of the rows, i.e. at the last row.
    Iterator beforeEnd ()
    {
      return Iterator(WrapperType(this),n-1);
    }

    //! @returns an iterator that is positioned before
    //! the first row of the matrix.
    Iterator beforeBegin ()
    {
      return Iterator(WrapperType(this),-1);
    }


    //! Iterator class for sequential access
    typedef ContainerWrapperIterator<const WrapperType, const_reference, const_reference> ConstIterator;
    //! typedef for stl compliant access
    typedef ConstIterator const_iterator;
    //! rename the iterators for easier access
    typedef ConstIterator ConstRowIterator;
    //! rename the iterators for easier access
    typedef typename const_row_type::ConstIterator ConstColIterator;

    //! begin iterator
    ConstIterator begin () const
    {
      return ConstIterator(WrapperType(this),0);
    }

    //! end iterator
    ConstIterator end () const
    {
      return ConstIterator(WrapperType(this),n);
    }

    //! @returns an iterator that is positioned before
    //! the end iterator of the rows. i.e. at the last row.
    ConstIterator beforeEnd() const
    {
      return ConstIterator(WrapperType(this),n-1);
    }

    //! @returns an iterator that is positioned before
    //! the first row of the matrix.
    ConstIterator beforeBegin () const
    {
      return ConstIterator(WrapperType(this),-1);
    }

    //===== vector space arithmetic

    //! vector space addition
    ScaledIdentityMatrix& operator+= (const ScaledIdentityMatrix& y)
    {
      p_ += y.p_;
      return *this;
    }

    //! vector space subtraction
    ScaledIdentityMatrix& operator-= (const ScaledIdentityMatrix& y)
    {
      p_ -= y.p_;
      return *this;
    }

    //! addition to the diagonal
    ScaledIdentityMatrix& operator+= (const K& k)
    {
      p_ += k;
      return *this;
    }

    //! subtraction from the diagonal
    ScaledIdentityMatrix& operator-= (const K& k)
    {
      p_ -= k;
      return *this;
    }
    //! vector space multiplication with scalar
    ScaledIdentityMatrix& operator*= (const K& k)
    {
      p_ *= k;
      return *this;
    }

    //! vector space division by scalar
    ScaledIdentityMatrix& operator/= (const K& k)
    {
      p_ /= k;
      return *this;
    }

    //===== binary operators

    //! vector space multiplication with scalar
    template <class Scalar,
              std::enable_if_t<IsNumber<Scalar>::value, int> = 0>
    friend auto operator* ( const ScaledIdentityMatrix& matrix, Scalar scalar)
    {
      return ScaledIdentityMatrix<typename PromotionTraits<K,Scalar>::PromotedType, n>{matrix.scalar()*scalar};
    }

    //! vector space multiplication with scalar
    template <class Scalar,
              std::enable_if_t<IsNumber<Scalar>::value, int> = 0>
    friend auto operator* (Scalar scalar, const ScaledIdentityMatrix& matrix)
    {
      return ScaledIdentityMatrix<typename PromotionTraits<Scalar,K>::PromotedType, n>{scalar*matrix.scalar()};
    }

    //===== comparison ops

    //! comparison operator
    bool operator==(const ScaledIdentityMatrix& other) const
    {
      return p_==other.scalar();
    }

    //! incomparison operator
    bool operator!=(const ScaledIdentityMatrix& other) const
    {
      return p_!=other.scalar();
    }

    //===== linear maps

    //! y = A x
    template<class X, class Y>
    void mv (const X& x, Y& y) const
    {
#ifdef DUNE_FMatrix_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(FMatrixError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(FMatrixError,"index out of range");
#endif
      for (size_type i=0; i<n; ++i)
        y[i] = p_ * x[i];
    }

    //! y = A^T x
    template<class X, class Y>
    void mtv (const X& x, Y& y) const
    {
      mv(x, y);
    }

    //! y += A x
    template<class X, class Y>
    void umv (const X& x, Y& y) const
    {
#ifdef DUNE_FMatrix_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(FMatrixError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(FMatrixError,"index out of range");
#endif
      for (size_type i=0; i<n; ++i)
        y[i] += p_ * x[i];
    }

    //! y += A^T x
    template<class X, class Y>
    void umtv (const X& x, Y& y) const
    {
#ifdef DUNE_FMatrix_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(FMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(FMatrixError,"index out of range");
#endif
      for (size_type i=0; i<n; ++i)
        y[i] += p_ * x[i];
    }

    //! y += A^H x
    template<class X, class Y>
    void umhv (const X& x, Y& y) const
    {
#ifdef DUNE_FMatrix_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(FMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(FMatrixError,"index out of range");
#endif
      for (size_type i=0; i<n; i++)
        y[i] += conjugateComplex(p_)*x[i];
    }

    //! y -= A x
    template<class X, class Y>
    void mmv (const X& x, Y& y) const
    {
#ifdef DUNE_FMatrix_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(FMatrixError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(FMatrixError,"index out of range");
#endif
      for (size_type i=0; i<n; ++i)
        y[i] -= p_ * x[i];
    }

    //! y -= A^T x
    template<class X, class Y>
    void mmtv (const X& x, Y& y) const
    {
#ifdef DUNE_FMatrix_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(FMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(FMatrixError,"index out of range");
#endif
      for (size_type i=0; i<n; ++i)
        y[i] -= p_ * x[i];
    }

    //! y -= A^H x
    template<class X, class Y>
    void mmhv (const X& x, Y& y) const
    {
#ifdef DUNE_FMatrix_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(FMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(FMatrixError,"index out of range");
#endif
      for (size_type i=0; i<n; i++)
        y[i] -= conjugateComplex(p_)*x[i];
    }

    //! y += alpha A x
    template<class X, class Y>
    void usmv (const K& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_FMatrix_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(FMatrixError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(FMatrixError,"index out of range");
#endif
      for (size_type i=0; i<n; i++)
        y[i] += alpha * p_ * x[i];
    }

    //! y += alpha A^T x
    template<class X, class Y>
    void usmtv (const K& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_FMatrix_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(FMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(FMatrixError,"index out of range");
#endif
      for (size_type i=0; i<n; i++)
        y[i] += alpha * p_ * x[i];
    }

    //! y += alpha A^H x
    template<class X, class Y>
    void usmhv (const K& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_FMatrix_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(FMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(FMatrixError,"index out of range");
#endif
      for (size_type i=0; i<n; i++)
        y[i] += alpha * conjugateComplex(p_) * x[i];
    }

    //===== norms

    //! frobenius norm: sqrt(sum over squared values of entries)
    typename FieldTraits<field_type>::real_type frobenius_norm () const
    {
      return fvmeta::sqrt(n*p_*p_);
    }

    //! square of frobenius norm, need for block recursion
    typename FieldTraits<field_type>::real_type frobenius_norm2 () const
    {
      return n*p_*p_;
    }

    //! infinity norm (row sum norm, how to generalize for blocks?)
    typename FieldTraits<field_type>::real_type infinity_norm () const
    {
      return std::abs(p_);
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    typename FieldTraits<field_type>::real_type infinity_norm_real () const
    {
      return fvmeta::absreal(p_);
    }

    //===== solve

    /** \brief Solve system A x = b
     */
    template<class V>
    void solve (V& x, const V& b) const
    {
      for (int i=0; i<n; i++)
        x[i] = b[i]/p_;
    }

    /** \brief Compute inverse
     */
    void invert()
    {
      p_ = 1/p_;
    }

    //! calculates the determinant of this matrix
    K determinant () const {
      return std::pow(p_,n);
    }

    //===== sizes

    //! number of blocks in row direction
    size_type N () const
    {
      return n;
    }

    //! number of blocks in column direction
    size_type M () const
    {
      return n;
    }

    //===== query

    //! return true when (i,j) is in pattern
    bool exists (size_type i, size_type j) const
    {
#ifdef DUNE_FMatrix_WITH_CHECKING
      if (i<0 || i>=n) DUNE_THROW(FMatrixError,"row index out of range");
      if (j<0 || j>=n) DUNE_THROW(FMatrixError,"column index out of range");
#endif
      return i==j;
    }

    //===== conversion operator

    /** \brief Sends the matrix to an output stream */
    friend std::ostream& operator<< (std::ostream& s, const ScaledIdentityMatrix<K,n>& a)
    {
      for (size_type i=0; i<n; i++) {
        for (size_type j=0; j<n; j++)
          s << ((i==j) ? a.p_ : 0) << " ";
        s << std::endl;
      }
      return s;
    }

    //! Return reference object as row replacement
    reference operator[](size_type i)
    {
      return reference(const_cast<K*>(&p_), i);
    }

    //! Return const_reference object as row replacement
    const_reference operator[](size_type i) const
    {
      return const_reference(const_cast<K*>(&p_), i);
    }

    //! Get const reference to diagonal entry
    const K& diagonal(size_type /*i*/) const
    {
      return p_;
    }

    //! Get reference to diagonal entry
    K& diagonal(size_type /*i*/)
    {
      return p_;
    }

    /** \brief Get const reference to the scalar diagonal value
     */
    const K& scalar() const
    {
      return p_;
    }

    /** \brief Get reference to the scalar diagonal value
     */
    K& scalar()
    {
      return p_;
    }

  private:
    // the data, very simply a single number
    K p_;

  };

  template <class DenseMatrix, class field, int N>
  struct DenseMatrixAssigner<DenseMatrix, ScaledIdentityMatrix<field, N>> {
    static void apply(DenseMatrix& denseMatrix,
                      ScaledIdentityMatrix<field, N> const& rhs) {
      assert(denseMatrix.M() == N);
      assert(denseMatrix.N() == N);
      denseMatrix = field(0);
      for (int i = 0; i < N; ++i)
        denseMatrix[i][i] = rhs.scalar();
    }
  };

  template<class K, int n>
  struct FieldTraits< ScaledIdentityMatrix<K, n> >
  {
    using field_type = typename ScaledIdentityMatrix<K, n>::field_type;
    using real_type = typename FieldTraits<field_type>::real_type;
  };

} // end namespace

#endif
