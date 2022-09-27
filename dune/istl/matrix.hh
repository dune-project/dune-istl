// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_MATRIX_HH
#define DUNE_ISTL_MATRIX_HH

/** \file
    \brief A dynamic dense block matrix class
 */

#include <cmath>
#include <memory>

#include <dune/common/ftraits.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/scalarvectorview.hh>
#include <dune/common/scalarmatrixview.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/istlexception.hh>
#include <dune/istl/blocklevel.hh>

namespace Dune {

namespace MatrixImp
{
  /**
      \brief A Vector of blocks with different blocksizes.

       This class started as a copy of VariableBlockVector, which used to be used for the internal memory managerment
       of the 'Matrix' class.  However, that mechanism stopped working when I started using the RandomAccessIteratorFacade
       in VariableBlockVector (308dd85483108f8baaa4051251e2c75e2a9aed32, to make VariableBlockVector pass a number of
       tightened interface compliance tests), and I couldn't quite figure out how to fix that.  However, using
       VariableBlockVector in Matrix internally was a hack anyway, so I simply took the working version of VariableBlockVector
       and copied it here under the new name of DenseMatrixBase.  This is still hacky, but one step closer to an
       elegant solution.
   */
  template<class B, class A=std::allocator<B> >
  class DenseMatrixBase : public Imp::block_vector_unmanaged<B,A>
                              // this derivation gives us all the blas level 1 and norms
                              // on the large array. However, access operators have to be
                              // overwritten.
  {
  public:

    //===== type definitions and constants

    //! export the type representing the field
    using field_type = typename Imp::BlockTraits<B>::field_type;

    //! export the allocator type
    typedef A allocator_type;

    //! The size type for the index access
    typedef typename A::size_type size_type;

    /** \brief Type of the elements of the outer vector, i.e., dynamic vectors of B
     *
     * Note that this is *not* the type referred to by the iterators and random access operators,
     * which return proxy objects.
     */
    typedef BlockVector<B,A> value_type;

    /** \brief Same as value_type, here for historical reasons
     */
    typedef BlockVector<B,A> block_type;

    // just a shorthand
    typedef Imp::BlockVectorWindow<B,A> window_type;

    typedef window_type reference;

    typedef const window_type const_reference;


    //===== constructors and such

    /** constructor without arguments makes empty vector,
            object cannot be used yet
     */
    DenseMatrixBase () : Imp::block_vector_unmanaged<B,A>()
    {
      // nothing is known ...
      rows_ = 0;
      columns_ = 0;
    }

    /** make vector with given number of blocks each having a constant size,
            object is fully usable then.

            \param _nblocks Number of blocks
            \param m Number of elements in each block
     */
    DenseMatrixBase (size_type rows, size_type columns) : Imp::block_vector_unmanaged<B,A>()
    {
      // and we can allocate the big array in the base class
      this->n = rows*columns;
      columns_ = columns;
      if (this->n>0)
      {
        this->p = allocator_.allocate(this->n);
        new (this->p)B[this->n];
      }
      else
      {
        this->n = 0;
        this->p = 0;
      }

      // we can allocate the windows now
      rows_ = rows;
    }

    //! copy constructor, has copy semantics
    DenseMatrixBase (const DenseMatrixBase& a)
    {
      // allocate the big array in the base class
      this->n = a.n;
      columns_ = a.columns_;
      if (this->n>0)
      {
        // allocate and construct objects
        this->p = allocator_.allocate(this->n);
        new (this->p)B[this->n];

        // copy data
        for (size_type i=0; i<this->n; i++)
          this->p[i]=a.p[i];
      }
      else
      {
        this->n = 0;
        this->p = nullptr;
      }

      // we can allocate the windows now
      rows_ = a.rows_;
    }

    //! free dynamic memory
    ~DenseMatrixBase ()
    {
      if (this->n>0) {
        size_type i=this->n;
        while (i)
          this->p[--i].~B();
        allocator_.deallocate(this->p,this->n);
      }
    }

    //! same effect as constructor with same argument
    void resize (size_type rows, size_type columns)
    {
      // deconstruct objects and deallocate memory if necessary
      if (this->n>0) {
        size_type i=this->n;
        while (i)
          this->p[--i].~B();
        allocator_.deallocate(this->p,this->n);
      }

      // and we can allocate the big array in the base class
      this->n = rows*columns;
      if (this->n>0)
      {
        this->p = allocator_.allocate(this->n);
        new (this->p)B[this->n];
      }
      else
      {
        this->n = 0;
        this->p = nullptr;
      }

      // we can allocate the windows now
      rows_ = rows;
      columns_ = columns;
    }

    //! assignment
    DenseMatrixBase& operator= (const DenseMatrixBase& a)
    {
      if (&a!=this)     // check if this and a are different objects
      {
        columns_ = a.columns_;
        // reallocate arrays if necessary
        // Note: still the block sizes may vary !
        if (this->n!=a.n || rows_!=a.rows_)
        {
          // deconstruct objects and deallocate memory if necessary
          if (this->n>0) {
            size_type i=this->n;
            while (i)
              this->p[--i].~B();
            allocator_.deallocate(this->p,this->n);
          }

          // allocate the big array in the base class
          this->n = a.n;
          if (this->n>0)
          {
            // allocate and construct objects
            this->p = allocator_.allocate(this->n);
            new (this->p)B[this->n];
          }
          else
          {
            this->n = 0;
            this->p = nullptr;
          }

          // Copy number of rows
          rows_ = a.rows_;
        }

        // and copy the data
        for (size_type i=0; i<this->n; i++)
          this->p[i]=a.p[i];
      }

      return *this;
    }


    //===== assignment from scalar

    //! assign from scalar
    DenseMatrixBase& operator= (const field_type& k)
    {
      (static_cast<Imp::block_vector_unmanaged<B,A>&>(*this)) = k;
      return *this;
    }


    //===== access to components
    // has to be overwritten from base class because it must
    // return access to the windows

    //! random access to blocks
    reference operator[] (size_type i)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (i>=rows_) DUNE_THROW(ISTLError,"index out of range");
#endif
      return window_type(this->p + i*columns_, columns_);
    }

    //! same for read only access
    const_reference operator[] (size_type i) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (i<0 || i>=rows_) DUNE_THROW(ISTLError,"index out of range");
#endif
      return window_type(this->p + i*columns_, columns_);
    }

    // forward declaration
    class ConstIterator;

    //! Iterator class for sequential access
    class Iterator
    {
    public:
      //! constructor, no arguments
      Iterator ()
      : window_(nullptr,0)
      {
        i = 0;
      }

      Iterator (Iterator& other) = default;
      Iterator (Iterator&& other) = default;

      //! constructor
      Iterator (B* data, size_type columns, size_type _i)
      : i(_i),
        window_(data + _i*columns, columns)
      {}

      /** \brief Move assignment */
      Iterator& operator=(Iterator&& other)
      {
        i = other.i;
        // Do NOT use window_.operator=, because that copies the window content, not just the window!
        window_.set(other.window_.getsize(),other.window_.getptr());
        return *this;
      }

      /** \brief Copy assignment */
      Iterator& operator=(Iterator& other)
      {
        i = other.i;
        // Do NOT use window_.operator=, because that copies the window content, not just the window!
        window_.set(other.window_.getsize(),other.window_.getptr());
        return *this;
      }

      //! prefix increment
      Iterator& operator++()
      {
        ++i;
        window_.setptr(window_.getptr()+window_.getsize());
        return *this;
      }

      //! prefix decrement
      Iterator& operator--()
      {
        --i;
        window_.setptr(window_.getptr()-window_.getsize());
        return *this;
      }

      //! equality
      bool operator== (const Iterator& it) const
      {
        return window_.getptr() == it.window_.getptr();
      }

      //! inequality
      bool operator!= (const Iterator& it) const
      {
        return window_.getptr() != it.window_.getptr();
      }

      //! equality
      bool operator== (const ConstIterator& it) const
      {
        return window_.getptr() == it.window_.getptr();
      }

      //! inequality
      bool operator!= (const ConstIterator& it) const
      {
        return window_.getptr() != it.window_.getptr();
      }

      //! dereferencing
      window_type& operator* () const
      {
        return window_;
      }

      //! arrow
      window_type* operator-> () const
      {
        return &window_;
      }

      // return index corresponding to pointer
      size_type index () const
      {
        return i;
      }

      friend class ConstIterator;

    private:
      size_type i;
      mutable window_type window_;
    };

    //! begin Iterator
    Iterator begin ()
    {
      return Iterator(this->p, columns_, 0);
    }

    //! end Iterator
    Iterator end ()
    {
      return Iterator(this->p, columns_, rows_);
    }

    //! @returns an iterator that is positioned before
    //! the end iterator of the vector, i.e. at the last entry.
    Iterator beforeEnd ()
    {
      return Iterator(this->p, columns_, rows_-1);
    }

    //! @returns an iterator that is positioned before
    //! the first entry of the vector.
    Iterator beforeBegin () const
    {
      return Iterator(this->p, columns_, -1);
    }

    //! random access returning iterator (end if not contained)
    Iterator find (size_type i)
    {
      return Iterator(this->p, columns_, std::min(i,rows_));
    }

    //! random access returning iterator (end if not contained)
    ConstIterator find (size_type i) const
    {
      return ConstIterator(this->p, columns_, std::min(i,rows_));
    }

    //! ConstIterator class for sequential access
    class ConstIterator
    {
    public:
      //! constructor
      ConstIterator ()
      : window_(nullptr,0)
      {
        i = 0;
      }

      //! constructor from pointer
      ConstIterator (const B* data, size_type columns, size_type _i)
      : i(_i),
        window_(const_cast<B*>(data + _i * columns), columns)
      {}

      //! constructor from non_const iterator
      ConstIterator (const Iterator& it)
      : i(it.i), window_(it.window_.getptr(),it.window_.getsize())
      {}

      ConstIterator& operator=(Iterator&& other)
      {
        i = other.i;
        // Do NOT use window_.operator=, because that copies the window content, not just the window!
        window_.set(other.window_.getsize(),other.window_.getptr());
        return *this;
      }

      ConstIterator& operator=(Iterator& other)
      {
        i = other.i;
        // Do NOT use window_.operator=, because that copies the window content, not just the window!
        window_.set(other.window_.getsize(),other.window_.getptr());
        return *this;
      }

      //! prefix increment
      ConstIterator& operator++()
      {
        ++i;
        window_.setptr(window_.getptr()+window_.getsize());
        return *this;
      }

      //! prefix decrement
      ConstIterator& operator--()
      {
        --i;
        window_.setptr(window_.getptr()-window_.getsize());
        return *this;
      }

      //! equality
      bool operator== (const ConstIterator& it) const
      {
        return window_.getptr() == it.window_.getptr();
      }

      //! inequality
      bool operator!= (const ConstIterator& it) const
      {
        return window_.getptr() != it.window_.getptr();
      }

      //! equality
      bool operator== (const Iterator& it) const
      {
        return window_.getptr() == it.window_.getptr();
      }

      //! inequality
      bool operator!= (const Iterator& it) const
      {
        return window_.getptr() != it.window_.getptr();
      }

      //! dereferencing
      const window_type& operator* () const
      {
        return window_;
      }

      //! arrow
      const window_type* operator-> () const
      {
        return &window_;
      }

      // return index corresponding to pointer
      size_type index () const
      {
        return i;
      }

      friend class Iterator;

    private:
      size_type i;
      mutable window_type window_;
    };

    /** \brief Export the iterator type using std naming rules */
    using iterator = Iterator;

    /** \brief Export the const iterator type using std naming rules */
    using const_iterator = ConstIterator;

    //! begin ConstIterator
    ConstIterator begin () const
    {
      return ConstIterator(this->p, columns_, 0);
    }

    //! end ConstIterator
    ConstIterator end () const
    {
      return ConstIterator(this->p, columns_, rows_);
    }

    //! @returns an iterator that is positioned before
    //! the end iterator of the vector. i.e. at the last element.
    ConstIterator beforeEnd() const
    {
      return ConstIterator(this->p, columns_, rows_-1);
    }

    //! end ConstIterator
    ConstIterator rend () const
    {
      return ConstIterator(this->p, columns_, -1);
    }

    //===== sizes

    //! number of blocks in the vector (are of variable size here)
    size_type N () const
    {
      return rows_;
    }


  private:
    size_type rows_;            // number of matrix rows
    size_type columns_;           // number of matrix columns

    A allocator_;
  };

}  // namespace MatrixImp

  /** \addtogroup ISTL_SPMV
     \{
   */
  /** \brief A generic dynamic dense matrix
   */
  template<class T, class A=std::allocator<T> >
  class Matrix
  {
  public:

    /** \brief Export the type representing the underlying field */
    using field_type = typename Imp::BlockTraits<T>::field_type;

    /** \brief Export the type representing the components */
    typedef T block_type;

    /** \brief Export the allocator */
    typedef A allocator_type;

    /** \brief The type implementing a matrix row */
    typedef typename MatrixImp::DenseMatrixBase<T,A>::window_type row_type;

    /** \brief Type for indices and sizes */
    typedef typename A::size_type size_type;

    /** \brief Iterator over the matrix rows */
    typedef typename MatrixImp::DenseMatrixBase<T,A>::Iterator RowIterator;

    /** \brief Iterator for the entries of each row */
    typedef typename row_type::iterator ColIterator;

    /** \brief Const iterator over the matrix rows */
    typedef typename MatrixImp::DenseMatrixBase<T,A>::ConstIterator ConstRowIterator;

    /** \brief Const iterator for the entries of each row */
    typedef typename row_type::const_iterator ConstColIterator;

    //! The number of nesting levels the matrix contains.
    [[deprecated("Use free function blockLevel(). Will be removed after 2.8.")]]
    static constexpr auto blocklevel = blockLevel<T>()+1;

    /** \brief Create empty matrix */
    Matrix() : data_(0,0), cols_(0)
    {}

    /** \brief Create uninitialized matrix of size rows x cols
     */
    Matrix(size_type rows, size_type cols) : data_(rows,cols), cols_(cols)
    {}

    /** \brief Change the matrix size
     *
     * The way the data is handled is unpredictable.
     */
    void setSize(size_type rows, size_type cols) {
      data_.resize(rows,cols);
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

    //! @returns an iterator that is positioned before
    //! the end iterator of the rows, i.e. at the last row.
    RowIterator beforeEnd ()
    {
      return data_.beforeEnd();
    }

    //! @returns an iterator that is positioned before
    //! the first row of the matrix.
    RowIterator beforeBegin ()
    {
      return data_.beforeBegin();
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

    //! @returns an iterator that is positioned before
    //! the end iterator of the rows. i.e. at the last row.
    ConstRowIterator beforeEnd() const
    {
      return data_.beforeEnd();
    }

    //! @returns an iterator that is positioned before
    //! the first row if the matrix.
    ConstRowIterator beforeBegin () const
    {
      return data_.beforeBegin();
    }

    /** \brief Assignment from scalar */
    Matrix& operator= (const field_type& t)
    {
      data_ = t;
      return *this;
    }

    /** \brief The index operator */
    row_type operator[](size_type row) {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (row<0)
        DUNE_THROW(ISTLError, "Can't access negative rows!");
      if (row>=N())
        DUNE_THROW(ISTLError, "Row index out of range!");
#endif
      return data_[row];
    }

    /** \brief The const index operator */
    const row_type operator[](size_type row) const {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (row<0)
        DUNE_THROW(ISTLError, "Can't access negative rows!");
      if (row>=N())
        DUNE_THROW(ISTLError, "Row index out of range!");
#endif
      return data_[row];
    }

    /** \brief Return the number of rows */
    size_type N() const {
      return data_.N();
    }

    /** \brief Return the number of columns */
    size_type M() const {
      return cols_;
    }

    /** \brief Multiplication with a scalar */
    Matrix<T>& operator*=(const field_type& scalar) {
      data_ *= scalar;
      return (*this);
    }

    /** \brief Division by a scalar */
    Matrix<T>& operator/=(const field_type& scalar) {
      data_ /= scalar;
      return (*this);
    }

    /*! \brief Add the entries of another matrix to this one.
     *
     * \param b The matrix to add to this one. Its size has to
     * be the same as the size of this matrix.
     */
    Matrix& operator+= (const Matrix& b) {
#ifdef DUNE_ISTL_WITH_CHECKING
      if(N()!=b.N() || M() != b.M())
        DUNE_THROW(RangeError, "Matrix sizes do not match!");
#endif
      data_ += b.data_;
      return (*this);
    }

    /*! \brief Subtract the entries of another matrix from this one.
     *
     * \param b The matrix to subtract from this one. Its size has to
     * be the same as the size of this matrix.
     */
    Matrix& operator-= (const Matrix& b) {
#ifdef DUNE_ISTL_WITH_CHECKING
      if(N()!=b.N() || M() != b.M())
        DUNE_THROW(RangeError, "Matrix sizes do not match!");
#endif
      data_ -= b.data_;
      return (*this);
    }

    /** \brief Return the transpose of the matrix */
    Matrix transpose() const {
      Matrix out(M(), N());
      for (size_type i=0; i<N(); i++)
        for (size_type j=0; j<M(); j++)
          out[j][i] = (*this)[i][j];

      return out;
    }

    /// Generic matrix multiplication.
    friend Matrix<T> operator*(const Matrix<T>& m1, const Matrix<T>& m2) {
      Matrix<T> out(m1.N(), m2.M());
      out = 0;

      for (size_type i=0; i<out.N(); i++ ) {
        for ( size_type j=0; j<out.M(); j++ )
          for (size_type k=0; k<m1.M(); k++)
            out[i][j] += m1[i][k]*m2[k][j];
      }

      return out;
    }

    /// Generic matrix-vector multiplication.
    template <class X, class Y>
    friend Y operator*(const Matrix<T>& m, const X& vec) {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (m.M()!=vec.size())
        DUNE_THROW(ISTLError, "Vector size doesn't match the number of matrix columns!");
#endif
      Y out(m.N());
      out = 0;

      for (size_type i=0; i<out.size(); i++ ) {
        for ( size_type j=0; j<vec.size(); j++ )
          out[i] += m[i][j]*vec[j];
      }

      return out;
    }

    //! y = A x
    template <class X, class Y>
    void mv(const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
      if (y.N()!=N()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
#endif
      for (size_type i=0; i<data_.N(); i++) {
        y[i]=0;
        for (size_type j=0; j<cols_; j++)
        {
          auto&& xj = Impl::asVector(x[j]);
          auto&& yi = Impl::asVector(y[i]);
          Impl::asMatrix((*this)[i][j]).umv(xj, yi);
        }
      }
    }

    //! y = A^T x
    template<class X, class Y>
    void mtv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
#endif
      for(size_type i=0; i<y.N(); ++i)
        y[i]=0;
      umtv(x,y);
    }

    //! y += A x
    template <class X, class Y>
    void umv(const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
      if (y.N()!=N()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
#endif
      for (size_type i=0; i<data_.N(); i++)
        for (size_type j=0; j<cols_; j++)
        {
          auto&& xj = Impl::asVector(x[j]);
          auto&& yi = Impl::asVector(y[i]);
          Impl::asMatrix((*this)[i][j]).umv(xj, yi);
        }
    }

    //! y -= A x
    template<class X, class Y>
    void mmv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
      if (y.N()!=N()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
#endif
      for (size_type i=0; i<data_.N(); i++)
        for (size_type j=0; j<cols_; j++)
        {
          auto&& xj = Impl::asVector(x[j]);
          auto&& yi = Impl::asVector(y[i]);
          Impl::asMatrix((*this)[i][j]).mmv(xj, yi);
        }
    }

    /** \brief \f$ y += \alpha A x \f$ */
    template <class X, class Y>
    void usmv(const field_type& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
      if (y.N()!=N()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
#endif
      for (size_type i=0; i<data_.N(); i++)
        for (size_type j=0; j<cols_; j++)
        {
          auto&& xj = Impl::asVector(x[j]);
          auto&& yi = Impl::asVector(y[i]);
          Impl::asMatrix((*this)[i][j]).usmv(alpha, xj, yi);
        }
    }

    //! y += A^T x
    template<class X, class Y>
    void umtv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
#endif
      for (size_type i=0; i<data_.N(); i++)
        for (size_type j=0; j<cols_; j++)
        {
          auto&& xi = Impl::asVector(x[i]);
          auto&& yj = Impl::asVector(y[j]);
          Impl::asMatrix((*this)[i][j]).umtv(xi, yj);
        }
    }

    //! y -= A^T x
    template<class X, class Y>
    void mmtv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
#endif
      for (size_type i=0; i<data_.N(); i++)
        for (size_type j=0; j<cols_; j++)
        {
          auto&& xi = Impl::asVector(x[i]);
          auto&& yj = Impl::asVector(y[j]);
          Impl::asMatrix((*this)[i][j]).mmtv(xi, yj);
        }
    }

    //! y += alpha A^T x
    template<class X, class Y>
    void usmtv (const field_type& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
#endif
      for (size_type i=0; i<data_.N(); i++)
        for (size_type j=0; j<cols_; j++)
        {
          auto&& xi = Impl::asVector(x[i]);
          auto&& yj = Impl::asVector(y[j]);
          Impl::asMatrix((*this)[i][j]).usmtv(alpha, xi, yj);
        }
    }

    //! y += A^H x
    template<class X, class Y>
    void umhv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
#endif
      for (size_type i=0; i<data_.N(); i++)
        for (size_type j=0; j<cols_; j++)
        {
          auto&& xi = Impl::asVector(x[i]);
          auto&& yj = Impl::asVector(y[j]);
          Impl::asMatrix((*this)[i][j]).umhv(xi,yj);
        }
    }

    //! y -= A^H x
    template<class X, class Y>
    void mmhv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
#endif
      for (size_type i=0; i<data_.N(); i++)
        for (size_type j=0; j<cols_; j++)
        {
          auto&& xi = Impl::asVector(x[i]);
          auto&& yj = Impl::asVector(y[j]);
          Impl::asMatrix((*this)[i][j]).mmhv(xi,yj);
        }
    }

    //! y += alpha A^H x
    template<class X, class Y>
    void usmhv (const field_type& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"vector/matrix size mismatch!");
#endif
      for (size_type i=0; i<data_.N(); i++)
        for (size_type j=0; j<cols_; j++)
        {
          auto&& xi = Impl::asVector(x[i]);
          auto&& yj = Impl::asVector(y[j]);
          Impl::asMatrix((*this)[i][j]).usmhv(alpha,xi,yj);
        }
    }

    //===== norms

    //! frobenius norm: sqrt(sum over squared values of entries)
    typename FieldTraits<field_type>::real_type frobenius_norm () const
    {
      return std::sqrt(frobenius_norm2());
    }

    //! square of frobenius norm, need for block recursion
    typename FieldTraits<field_type>::real_type frobenius_norm2 () const
    {
      typename FieldTraits<field_type>::real_type sum=0;
      for (size_type i=0; i<this->N(); i++)
        for (size_type j=0; j<this->M(); j++)
          sum += Impl::asMatrix(data_[i][j]).frobenius_norm2();
      return sum;
    }

    //! infinity norm (row sum norm, how to generalize for blocks?)
    template <typename ft = field_type,
              typename std::enable_if<!HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm() const {
      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;

      real_type norm = 0;
      for (auto const &x : *this) {
        real_type sum = 0;
        for (auto const &y : x)
          sum += Impl::asMatrix(y).infinity_norm();
        norm = max(sum, norm);
        isNaN += sum;
      }

      return norm;
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    template <typename ft = field_type,
              typename std::enable_if<!HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm_real() const {
      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;

      real_type norm = 0;
      for (auto const &x : *this) {
        real_type sum = 0;
        for (auto const &y : x)
          sum += Impl::asMatrix(y).infinity_norm_real();
        norm = max(sum, norm);
      }
      return norm;
    }

    //! infinity norm (row sum norm, how to generalize for blocks?)
    template <typename ft = field_type,
              typename std::enable_if<HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm() const {
      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;

      real_type norm = 0;
      real_type isNaN = 1;
      for (auto const &x : *this) {
        real_type sum = 0;
        for (auto const &y : x)
          sum += Impl::asMatrix(y).infinity_norm();
        norm = max(sum, norm);
        isNaN += sum;
      }

      return norm * (isNaN / isNaN);
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    template <typename ft = field_type,
              typename std::enable_if<HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm_real() const {
      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;

      real_type norm = 0;
      real_type isNaN = 1;
      for (auto const &x : *this) {
        real_type sum = 0;
        for (auto const &y : x)
          sum += Impl::asMatrix(y).infinity_norm_real();
        norm = max(sum, norm);
        isNaN += sum;
      }

      return norm * (isNaN / isNaN);
    }

    //===== query

    //! return true if (i,j) is in pattern
    bool exists ([[maybe_unused]] size_type i, [[maybe_unused]] size_type j) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (i<0 || i>=N()) DUNE_THROW(ISTLError,"row index out of range");
      if (j<0 || i>=M()) DUNE_THROW(ISTLError,"column index out of range");
#endif
      return true;
    }

  protected:

    /** \brief Abuse DenseMatrixBase as an engine for a 2d array ISTL-style
     */
    MatrixImp::DenseMatrixBase<T,A> data_;

    /** \brief Number of columns of the matrix

       In general you can extract the same information from the data_ member.  However if you
       want to be able to properly handle 0xn matrices then you need a separate member.
     */
    size_type cols_;
  };

  template<class T, class A>
  struct FieldTraits< Matrix<T, A> >
  {
    using field_type = typename Matrix<T, A>::field_type;
    using real_type = typename FieldTraits<field_type>::real_type;
  };

  /** \} */
} // end namespace Dune

#endif
