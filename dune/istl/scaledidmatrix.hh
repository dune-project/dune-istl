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

  /** @brief A multiple of the identity matrix of static size
   *
   * This class provides all operations specified by the dune-istl matrix API.
   * The implementations exploit the fact that the matrix is a multiple of the identity.
   *
   * \tparam K The number used for matrix entries
   * \tparam n The number of matrix rows and columns
   *
   * \warning If `m` is an object of type `ScaledIdentityMatrix`, then code like
   * ~~~{.cpp}
   * K entry = m[0][1];
   * ~~~
   * will compile.  However, the expression `m[0][1]` is *not*
   * guaranteed to return 0. Rather, its behavior is undefined.
   */
  template<class K, int n>
  class ScaledIdentityMatrix
  {
    typedef DiagonalMatrixWrapper< ScaledIdentityMatrix<K,n> > WrapperType;

  public:
    //===== type definitions and constants

    //! The type representing numbers
    typedef K field_type;

    //! The type representing matrix entries (which may be matrices themselves)
    typedef K block_type;

    //! The type used for the indices and sizes
    typedef std::size_t size_type;

    /** \brief Type of a single matrix row
     *
     * Since the implementation does not store actual rows,
     * this is a proxy type, which tries to behave like an array
     * of matrix entries as much as possible.
     * \note The type is really `DiagonalRowVector`. Implementing a
     * dedicated `ScaledIdentityMatrixRowVector` would just be a copy of that.
     */
    typedef DiagonalRowVector<K,n> row_type;
    typedef row_type reference;

    /** \brief Const type of a single row */
    typedef DiagonalRowVectorConst<K,n> const_row_type;
    typedef const_row_type const_reference;

    //! The number of matrix rows
    static constexpr std::integral_constant<size_type,n> rows = {};

    //! The number of matrix columns
    static constexpr std::integral_constant<size_type,n> cols = {};

    //===== constructors
    /** \brief Default constructor
     */
    ScaledIdentityMatrix () {}

    /** \brief Constructor initializing the whole matrix with a scalar
     */
    ScaledIdentityMatrix (const K& k)
      : p_(k)
    {}

    /** \brief Assignment from scalar
     */
    ScaledIdentityMatrix& operator= (const K& k)
    {
      p_ = k;
      return *this;
    }

    /** \brief Check if matrix is identical to other matrix
     *
     * "Identical" means: Not just the same values, but the very same object.
     */
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

    /** @name Vector space arithmetic
     */
    ///@{

    //! Vector space addition
    ScaledIdentityMatrix& operator+= (const ScaledIdentityMatrix& y)
    {
      p_ += y.p_;
      return *this;
    }

    //! Vector space subtraction
    ScaledIdentityMatrix& operator-= (const ScaledIdentityMatrix& y)
    {
      p_ -= y.p_;
      return *this;
    }

    /** \brief Addition of a scalar to the diagonal
     *
     * \warning This is not the same as adding a scalar to a FieldMatrix!
     */
    ScaledIdentityMatrix& operator+= (const K& k)
    {
      p_ += k;
      return *this;
    }

    /** \brief Subtraction of a scalar from the diagonal
     *
     * \warning This is not the same as subtracting a scalar from a FieldMatrix!
     */
    ScaledIdentityMatrix& operator-= (const K& k)
    {
      p_ -= k;
      return *this;
    }

    //! Vector space multiplication with scalar
    ScaledIdentityMatrix& operator*= (const K& k)
    {
      p_ *= k;
      return *this;
    }

    //! Vector space division by scalar
    ScaledIdentityMatrix& operator/= (const K& k)
    {
      p_ /= k;
      return *this;
    }

    //! Vector space multiplication with scalar
    template <class Scalar,
              std::enable_if_t<IsNumber<Scalar>::value, int> = 0>
    friend auto operator* ( const ScaledIdentityMatrix& matrix, Scalar scalar)
    {
      return ScaledIdentityMatrix<typename PromotionTraits<K,Scalar>::PromotedType, n>{matrix.scalar()*scalar};
    }

    //! Vector space multiplication with scalar
    template <class Scalar,
              std::enable_if_t<IsNumber<Scalar>::value, int> = 0>
    friend auto operator* (Scalar scalar, const ScaledIdentityMatrix& matrix)
    {
      return ScaledIdentityMatrix<typename PromotionTraits<Scalar,K>::PromotedType, n>{scalar*matrix.scalar()};
    }

    //! Addition of ScaledIdentityMatrix to FieldMatrix
    template <class OtherScalar>
      requires requires(K k, OtherScalar otherScalar) { k + otherScalar; }
    friend auto& operator+= (FieldMatrix<OtherScalar,n,n>& fieldMatrix,
                             const ScaledIdentityMatrix& matrix)
    {
      for (int i=0; i<n; i++)
        fieldMatrix[i][i] += matrix.p_;

      return fieldMatrix;
    }

    //! Addition of ScaledIdentityMatrix to FieldMatrix
    template <class OtherScalar>
      requires requires(K k, OtherScalar otherScalar) { k + otherScalar; }
    friend auto operator+ (const FieldMatrix<OtherScalar,n,n>& fieldMatrix,
                           const ScaledIdentityMatrix& matrix)
    {
      using Result = FieldMatrix<typename PromotionTraits<K,OtherScalar>::PromotedType,n,n>;
      Result result = fieldMatrix;
      result += matrix;
      return result;
    }

    //! Addition of FieldMatrix to ScaledIdentityMatrix
    template <class OtherScalar>
      requires requires(K k, OtherScalar otherScalar) { k + otherScalar; }
    friend auto operator+ (const ScaledIdentityMatrix& matrix,
                           const FieldMatrix<OtherScalar,n,n>& fieldMatrix)
    {
      return fieldMatrix + matrix;
    }

    //! Addition of ScaledIdentityMatrix to DiagonalMatrix
    template <class OtherScalar>
      requires requires(K k, OtherScalar otherScalar) { k + otherScalar; }
    friend auto operator+= (DiagonalMatrix<OtherScalar,n>& diagonalMatrix,
                            const ScaledIdentityMatrix& matrix)
    {
      for (std::size_t i=0; i<n; i++)
        diagonalMatrix.diagonal(i) += matrix.p_;

      return diagonalMatrix;
    }

    //! Addition of ScaledIdentityMatrix to DiagonalMatrix
    template <class OtherScalar>
      requires requires(K k, OtherScalar otherScalar) { k + otherScalar; }
    friend auto operator+ (const DiagonalMatrix<OtherScalar,n>& diagonalMatrix,
                           const ScaledIdentityMatrix& matrix)
    {
      using Result = DiagonalMatrix<typename PromotionTraits<K,OtherScalar>::PromotedType,n>;
      Result result = diagonalMatrix;
      result += matrix;
      return result;
    }

    //! Addition of DiagonalMatrix to ScaledIdentityMatrix
    template <class OtherScalar>
      requires requires(K k, OtherScalar otherScalar) { k + otherScalar; }
    friend auto operator+ (const ScaledIdentityMatrix& matrix,
                           const DiagonalMatrix<OtherScalar,n>& diagonalMatrix)
    {
      return diagonalMatrix + matrix;
    }
    ///@}   // Vector space arithmetic

    //===== comparison ops

    //! Test for equality
    bool operator==(const ScaledIdentityMatrix& other) const
    {
      return p_==other.scalar();
    }

    //! Test for inequality
    bool operator!=(const ScaledIdentityMatrix& other) const
    {
      return p_!=other.scalar();
    }

    /** \name Matrix-vector products
     */
    ///@{

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

    ///@}  // Matrix-vector products


    /** @name Norms
     */
    ///@{

    //! The Frobenius norm
    typename FieldTraits<field_type>::real_type frobenius_norm () const
    {
      return fvmeta::sqrt(n*p_*p_);
    }

    //! The square of the Frobenius norm
    typename FieldTraits<field_type>::real_type frobenius_norm2 () const
    {
      return n*p_*p_;
    }

    /** \brief The row sum norm
     *
     * For a multiple of the identity matrix, this is simply the absolute value
     * of any diagonal matrix entry.
     */
    typename FieldTraits<field_type>::real_type infinity_norm () const
    {
      return std::abs(p_);
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    typename FieldTraits<field_type>::real_type infinity_norm_real () const
    {
      return fvmeta::absreal(p_);
    }

    ///@}

    /** \brief Solve linear system A x = b
     *
     * \tparam V Vector data type
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

    //! Calculates the determinant of this matrix
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

    //! Return true if (i,j) is in the matrix pattern, i.e., if i==j
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
