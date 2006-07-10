// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_MATRIX_HH
#define DUNE_MATRIX_HH

/** \file
    \brief A dynamic dense block matrix class
 */

#include <vector>
#include <dune/istl/bvector.hh>

namespace Dune {

  /** \brief A generic dynamic matrix
      \addtogroup ISTL
   */
  template<class T, class A=ISTLAllocator>
  class Matrix
  {
  public:

    /** \brief Export the type representing the underlying field */
    typedef typename T::field_type field_type;

    /** \brief Export the type representing the components */
    typedef T block_type;

    /** \brief Export the allocator */
    typedef A allocator_type;

    /** \brief The type implementing a matrix row */
    typedef BlockVector<T> row_type;

    /** \brief Type for indices and sizes */
    typedef typename A::size_type size_type;

    /** \brief Iterator over the matrix rows */
    typedef typename BlockVector<row_type>::iterator RowIterator;

    /** \brief Iterator for the entries of each row */
    typedef typename row_type::iterator ColIterator;

    /** \brief Const iterator over the matrix rows */
    typedef typename BlockVector<row_type>::const_iterator ConstRowIterator;

    /** \brief Const iterator for the entries of each row */
    typedef typename row_type::const_iterator ConstColIterator;

    /** \brief Create empty matrix */
    Matrix() : data_(0), cols_(0)
    {}

    /** \brief Create uninitialized matrix of size rows x cols
     */
    Matrix(int rows, int cols) : data_(rows), cols_(cols)
    {
      for (int i=0; i<rows; i++)
        data_[i].resize(cols);
    }

    /** \brief Change the matrix size
     *
     * The way the data is handled is unpredictable.
     */
    void resize(int rows, int cols) {
      data_.resize(rows);
      for (int i=0; i<rows; i++)
        data_[i].resize(cols);
      cols_ = cols;
    }

    /** \brief Get iterator to first row */
    RowIterator begin()
    {
      return data_.begin();
    }

    /** \brief Get iterator to one beyond last row */
    RowIterator end()
    {
      return data_.end();
    }

    /** \brief Get iterator to last row */
    RowIterator rbegin()
    {
      return data_.rbegin();
    }

    /** \brief Get iterator to one before first row */
    RowIterator rend()
    {
      return data_.rend();
    }

    /** \brief Get const iterator to first row */
    ConstRowIterator begin() const
    {
      return data_.begin();
    }

    /** \brief Get const iterator to one beyond last row */
    ConstRowIterator end() const
    {
      return data_.end();
    }

    /** \brief Get const iterator to last row */
    ConstRowIterator rbegin() const
    {
      return data_.rbegin();
    }

    /** \brief Get const iterator to one before first row */
    ConstRowIterator rend() const
    {
      return data_.rend();
    }

    /** \brief Assignment from scalar */
    Matrix& operator= (const field_type& t)
    {
      for (unsigned int i=0; i<data_.size(); i++)
        data_[i] = t;
    }

    /** \brief The index operator */
    row_type& operator[](int row) {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (row<0)
        DUNE_THROW(ISTLError, "Can't access negative rows!");
      if (row>=rows_)
        DUNE_THROW(ISTLError, "Row index out of range!");
#endif
      return data_[row];
    }

    /** \brief The const index operator */
    const row_type& operator[](int row) const {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (row<0)
        DUNE_THROW(ISTLError, "Can't access negative rows!");
      if (row>=rows_)
        DUNE_THROW(ISTLError, "Row index out of range!");
#endif
      return data_[row];
    }

    /** \brief Return the number of rows */
    size_type N() const {
      return data_.size();
    }

    /** \brief Return the number of columns */
    size_type M() const {
      return cols_;
    }

    /** \brief The number of scalar rows */
    size_type rowdim() const {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (M()==0)
        DUNE_THROW(ISTLError, "Can't compute rowdim() when there are no columns!");
#endif
      size_type dim = 0;
      for (int i=0; i<data_.size(); i++)
        dim += data_[i][0].rowdim();

      return dim;
    }

    /** \brief The number of scalar columns */
    size_type coldim() const {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (N()==0)
        DUNE_THROW(ISTLError, "Can't compute coldim() when there are no rows!");
#endif
      size_type dim = 0;
      for (int i=0; i<data_[0].size(); i++)
        dim += data_[0][i].coldim();

      return dim;
    }

    /** \brief The number of scalar rows */
    size_type rowdim(size_type r) const {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (r<0 || r>=N())
        DUNE_THROW(ISTLError, "Rowdim for nonexisting row " << r << " requested!");
      if (M()==0)
        DUNE_THROW(ISTLError, "Can't compute rowdim() when there are no columns!");
#endif
      return data_[r][0].rowdim();
    }

    /** \brief The number of scalar columns */
    size_type coldim(size_type c) const {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (c<0 || c>=M())
        DUNE_THROW(ISTLError, "Coldim for nonexisting column " << c << " requested!");
      if (N()==0)
        DUNE_THROW(ISTLError, "Can't compute coldim() when there are no rows!");
#endif
      return data_[0][c].coldim();
    }

    /** \brief Multiplication with a scalar */
    Matrix<T> operator*=(const T& scalar) {
      for (int row=0; row<data_.size(); row++)
        for (int col=0; col<cols_; col++)
          (*this)[row][col] *= scalar;

      return (*this);
    }

    /** \brief Return the transpose of the matrix */
    Matrix transpose() const {
      Matrix out(N(), M());
      for (int i=0; i<M(); i++)
        for (int j=0; j<N(); j++)
          out[j][i] = (*this)[i][j];

      return out;
    }

    //! Multiplication of the transposed matrix times a vector
    template <class X, class Y>
    Y transposedMult(const Y& vec) {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (N()!=vec.size())
        DUNE_THROW(ISTLError, "Vector size doesn't match the number of matrix rows!");
#endif
      Y out(M());
      out = 0;

      for (int i=0; i<out.size(); i++ ) {
        for ( int j=0; j<vec.size(); j++ )
          out[i] += (*this)[j][i]*vec[j];
      }

      return out;
    }

    /// Generic matrix multiplication.
    friend Matrix<T> operator*(const Matrix<T>& m1, const Matrix<T>& m2) {
      Matrix<T> out(m1.N(), m2.M());
      out.clear();

      for (int i=0; i<out.N(); i++ ) {
        for ( int j=0; j<out.M(); j++ )
          for (int k=0; k<m1.M(); k++)
            out[i][j] += m1[i][k]*m2[k][j];
      }

      return out;
    }

    /// Generic matrix-vector multiplication.
    template <class X, class Y>
    friend Y operator*(const Matrix<T>& m, const X& vec) {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (M()!=vec.size())
        DUNE_THROW(ISTLError, "Vector size doesn't match the number of matrix columns!");
#endif
      Y out(m.N());
      out = 0;

      for (int i=0; i<out.size(); i++ ) {
        for ( int j=0; j<vec.size(); j++ )
          out[i] += m[i][j]*vec[j];
      }

      return out;
    }

    //! y += A x
    template <class X, class Y>
    void umv(const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
#endif

      for (int i=0; i<data_.size(); i++) {

        for (int j=0; j<cols_; j++)
          (*this)[i][j].umv(x[j], y[i]);

      }

    }

    /** \brief \f$ y += \alpha A x \f$ */
    template <class X, class Y>
    void usmv(const field_type& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
#endif

      for (int i=0; i<data_.size(); i++) {

        for (int j=0; j<cols_; j++)
          (*this)[i][j].usmv(alpha, x[j], y[i]);

      }

    }

  protected:

    BlockVector<row_type> data_;

    int cols_;
  };

} // end namespace Dune

#endif
