// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_SCALED_IDENTITY_MATRIX_HH
#define DUNE_SCALED_IDENTITY_MATRIX_HH

/*! \file

   \brief  This file implements a quadratic matrix of fixed size which is
   a multiple of the identity.
 */

#include <cmath>
#include <cstddef>
#include <complex>
#include <iostream>
#include <dune/common/exceptions.hh>

namespace Dune {

  /**
      @brief A multiple of the identity matrix of static size
   */
  template<class K, int n>
  class ScaledIdentityMatrix
  {

  public:
    //===== type definitions and constants

    //! export the type representing the field
    typedef K field_type;

    //! export the type representing the components
    typedef K block_type;

    //! The type used for the index access and size operations.
    typedef std::size_t size_type;

    //! We are at the leaf of the block recursion
    enum {
      //! The number of block levels we contain. This is 1.
      blocklevel = 1
    };

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

    //! vector space multiplication with scalar
    ScaledIdentityMatrix& operator+= (const K& k)
    {
      p_ += k;
      return *this;
    }

    //! vector space division by scalar
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
        y[i] += fm_ck(p_)*x[i];
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
        y[i] -= fm_ck(p_)*x[i];
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
        y[i] += alpha * fm_ck(p_) * x[i];
    }

    //===== norms

    //! frobenius norm: sqrt(sum over squared values of entries)
    double frobenius_norm () const
    {
      return std::sqrt(n*p_*p_);
    }

    //! square of frobenius norm, need for block recursion
    double frobenius_norm2 () const
    {
      return n*p_*p_;
    }

    //! infinity norm (row sum norm, how to generalize for blocks?)
    double infinity_norm () const
    {
      return std::fabs(p_);
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    double infinity_norm_real () const
    {
      return fvmeta_absreal(p_);
    }

    //===== solve

    /** \brief Solve system A x = b
     */
    template<class V>
    void solve (V& x, const V& b) const {
      for (int i=0; i<n; i++)
        x = b/p_;
    }

    /** \brief Compute inverse
     */
    void invert() {
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
      if (j<0 || j>=m) DUNE_THROW(FMatrixError,"column index out of range");
#endif
      return i==j;
    }

    //===== conversion operator

    /** \brief Cast to a scalar */
    //        operator K () const {return p_;}

    /** \brief Cast to FieldMatrix
     * Might be inefficient, but operator= has to be a member of FieldMatrix
     * */
    operator FieldMatrix<K,n,n>() const
    {
      FieldMatrix<K, n, n> fm = 0.0;
      for(int i=0; i<n; ++i)
        fm[i][i] = p_;
      return fm;
    }



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

    /** \brief Return FieldVector as row replacement
     * This might be inefficient.
     * */
    const FieldVector<K,n> operator[](size_type i) const
    {
      FieldVector<K, n> fv;
      fv = 0.0;
      fv[i] = p_;
      return fv;
    }


  private:
    // the data, very simply a single number
    K p_;

  };

} // end namespace

#endif
