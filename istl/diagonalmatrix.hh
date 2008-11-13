// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_DIAGONAL_MATRIX_HH
#define DUNE_DIAGONAL_MATRIX_HH

/*! \file
   \brief  This file implements a quadratic matrix of fixed size which is diagonal.
 */

#include <cmath>
#include <cstddef>
#include <complex>
#include <iostream>
#include <dune/common/exceptions.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/genericiterator.hh>



namespace Dune {

  template< class K, int n > class DiagonalRowVectorConst;
  template< class K, int n > class DiagonalRowVector;
  template< class C, class T, class R, class S> class ReferenceStorageIterator;


  /**
      @brief A diagonal matrix of static size
   */
  template<class K, int n>
  class DiagonalMatrix
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

    //! Default constructor
    DiagonalMatrix () {}

    //! Constructor initializing the whole matrix with a scalar
    DiagonalMatrix (const K& k)
      : diag_(k)
    {}

    //! Constructor initializing the diagonal with a vector
    DiagonalMatrix (const FieldVector<K,n>& diag)
      : diag_(diag)
    {}


    //===== assignment from scalar
    DiagonalMatrix& operator= (const K& k)
    {
      diag_ = k;
      return *this;
    }

    // check if matrix is identical to other matrix (not only identical values)
    bool identical(const DiagonalMatrix<K,n>& other) const
    {
      return (this==&other);
    }

    //===== iterator interface to rows of the matrix
    //! Iterator class for sequential access
    typedef ReferenceStorageIterator<DiagonalMatrix<K,n>,reference,reference,DiagonalMatrix<K,n>&> Iterator;
    //! typedef for stl compliant access
    typedef Iterator iterator;
    //! rename the iterators for easier access
    typedef Iterator RowIterator;
    //! rename the iterators for easier access
    typedef typename row_type::Iterator ColIterator;

    //! begin iterator
    Iterator begin ()
    {
      return Iterator(*this,0);
    }

    //! end iterator
    Iterator end ()
    {
      return Iterator(*this,n);
    }

    //! begin iterator
    Iterator rbegin ()
    {
      return Iterator(*this,n-1);
    }

    //! end iterator
    Iterator rend ()
    {
      return Iterator(*this,-1);
    }


    //! Iterator class for sequential access
    typedef ReferenceStorageIterator<const DiagonalMatrix<K,n>,const_reference,const_reference,const DiagonalMatrix<K,n>&> ConstIterator;
    //! typedef for stl compliant access
    typedef ConstIterator const_iterator;
    //! rename the iterators for easier access
    typedef ConstIterator ConstRowIterator;
    //! rename the iterators for easier access
    typedef typename const_row_type::ConstIterator ConstColIterator;

    //! begin iterator
    ConstIterator begin () const
    {
      return ConstIterator(*this,0);
    }

    //! end iterator
    ConstIterator end () const
    {
      return ConstIterator(*this,n);
    }

    //! begin iterator
    ConstIterator rbegin () const
    {
      return ConstIterator(*this,n-1);
    }

    //! end iterator
    ConstIterator rend () const
    {
      return ConstIterator(*this,-1);
    }



    //===== vector space arithmetic

    //! vector space addition
    DiagonalMatrix& operator+= (const DiagonalMatrix& y)
    {
      diag_ += y.diag_;
      return *this;
    }

    //! vector space subtraction
    DiagonalMatrix& operator-= (const DiagonalMatrix& y)
    {
      diag_ -= y.diag_;
      return *this;
    }

    //! vector space multiplication with scalar
    DiagonalMatrix& operator+= (const K& k)
    {
      diag_ += k;
      return *this;
    }

    //! vector space division by scalar
    DiagonalMatrix& operator-= (const K& k)
    {
      diag_ -= k;
      return *this;
    }

    //! vector space multiplication with scalar
    DiagonalMatrix& operator*= (const K& k)
    {
      diag_ *= k;
      return *this;
    }

    //! vector space division by scalar
    DiagonalMatrix& operator/= (const K& k)
    {
      diag_ /= k;
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
        y[i] = diag_[i] * x[i];
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
        y[i] += diag_[i] * x[i];
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
        y[i] += diag_[i] * x[i];
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
        y[i] += fm_ck(diag_[i])*x[i];
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
        y[i] -= diag_[i] * x[i];
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
        y[i] -= diag_[i] * x[i];
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
        y[i] -= fm_ck(diag_[i])*x[i];
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
        y[i] += alpha * diag_[i] * x[i];
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
        y[i] += alpha * diag_[i] * x[i];
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
        y[i] += alpha * fm_ck(diag_[i]) * x[i];
    }

    //===== norms

    //! frobenius norm: sqrt(sum over squared values of entries)
    double frobenius_norm () const
    {
      return diag_.two_norm();
    }

    //! square of frobenius norm, need for block recursion
    double frobenius_norm2 () const
    {
      return diag_.two_norm2();
    }

    //! infinity norm (row sum norm, how to generalize for blocks?)
    double infinity_norm () const
    {
      return diag_.infinity_norm();
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    double infinity_norm_real () const
    {
      return diag_.infinity_norm_real();
    }



    //===== solve

    //! Solve system A x = b
    template<class V>
    void solve (V& x, const V& b) const
    {
      for (int i=0; i<n; i++)
        x[i] = b[i]/diag_[i];
    }

    //! Compute inverse
    void invert()
    {
      for (int i=0; i<n; i++)
        diag_[i] = 1/diag_[i];
    }

    //! calculates the determinant of this matrix
    K determinant () const
    {
      K det = diag_[0];
      for (int i=1; i<n; i++)
        det *= diag_[i];
      return det;
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



    //! Sends the matrix to an output stream
    friend std::ostream& operator<< (std::ostream& s, const DiagonalMatrix<K,n>& a)
    {
      for (size_type i=0; i<n; i++) {
        for (size_type j=0; j<n; j++)
          s << ((i==j) ? a.diag_[i] : 0) << " ";
        s << std::endl;
      }
      return s;
    }

    //! Return reference object as row replacement
    reference operator[](size_type i)
    {
      return reference(const_cast<K*>(&diag_[i]), i);
    }

    //! Return const_reference object as row replacement
    const_reference operator[](size_type i) const
    {
      return const_reference(const_cast<K*>(&diag_[i]), i);
    }

    //! Get const reference to diagonal entry
    const K& diagonal(size_type i) const
    {
      return diag_[i];
    }

    //! Get reference to diagonal entry
    K& diagonal(size_type i)
    {
      return diag_[i];
    }

    //! Get const reference to diagonal vector
    const FieldVector<K,n>& diagonal() const
    {
      return diag_;
    }

    //! Get reference to diagonal vector
    FieldVector<K,n>& diagonal()
    {
      return diag_;
    }

  private:

    // the data, a FieldVector storing the diagonal
    FieldVector<K,n> diag_;
  };


  /** \brief
   *
   */
  template< class K, int n >
  class DiagonalRowVectorConst
  {
  public:
    // remember size of vector
    enum { dimension = n };

    // standard constructor and everything is sufficient ...

    //===== type definitions and constants

    //! export the type representing the field
    typedef K field_type;

    //! export the type representing the components
    typedef K block_type;

    //! The type used for the index access and size operation
    typedef std::size_t size_type;

    //! We are at the leaf of the block recursion
    enum {
      //! The number of block levels we contain
      blocklevel = 1
    };

    //! export size
    enum {
      //! The size of this vector.
      size = n
    };

    //! Constructor making uninitialized vector
    DiagonalRowVectorConst() :
      p_(0),
      row_(0)
    {}

    //! Constructor making vector with identical coordinates
    explicit DiagonalRowVectorConst (K* p, int col) :
      p_(p),
      row_(col)
    {}

    //===== access to components

    //! same for read only access
    const K& operator[] (size_type i) const
    {
      if (i!=row_)
        DUNE_THROW(FMatrixError,"index is read only");
      //            return ZERO<K>::Value;
      return *p_;
    }

    // check if row is identical to other row (not only identical values)
    // since this is a proxy class we need to check equality of the stored pointer
    bool identical(const DiagonalRowVectorConst<K,n>& other) const
    {
      return ((p_ == other.p_)and (row_ == other.row_));
    }

    //! ConstIterator class for sequential access
    typedef ReferenceStorageIterator<DiagonalRowVectorConst<K,n>,const K,const K&,DiagonalRowVectorConst<K,n> > ConstIterator;
    //! typedef for stl compliant access
    typedef ConstIterator const_iterator;

    //! begin ConstIterator
    ConstIterator begin () const
    {
      // the iterator gets this type as '[...]Const' and not as 'const [...]'
      // thus we must remove constness
      return ConstIterator(*const_cast<DiagonalRowVectorConst<K,n>*>(this),row_);
    }

    //! end ConstIterator
    ConstIterator end () const
    {
      // the iterator gets this type as '[...]Const' and not as 'const [...]'
      // thus we must remove constness
      return ConstIterator(*const_cast<DiagonalRowVectorConst<K,n>*>(this),row_+1);
    }

    //! begin ConstIterator
    ConstIterator rbegin () const
    {
      // the iterator gets this type as '[...]Const' and not as 'const [...]'
      // thus we must remove constness
      return ConstIterator(*const_cast<DiagonalRowVectorConst<K,n>*>(this),row_);
    }

    //! end ConstIterator
    ConstIterator rend () const
    {
      // the iterator gets this type as '[...]Const' and not as 'const [...]'
      // thus we must remove constness
      return ConstIterator(*const_cast<DiagonalRowVectorConst<K,n>*>(this),-1);
    }

    //! return iterator to given element or end()
    //    ConstIterator find (size_type i) const
    //    {
    //        if (i<n)
    //            return ConstIterator(*this,i);
    //        else
    //            return ConstIterator(*this,n);
    //    }

    //! Binary vector comparison
    bool operator== (const DiagonalRowVectorConst& y) const
    {
      return ((p_==y.p_)and (row_==y.row_));
    }

    //===== sizes

    //! number of blocks in the vector (are of size 1 here)
    size_type N () const
    {
      return n;
    }

    //! dimension of the vector space
    size_type dim () const
    {
      return n;
    }

    //! index of this row in surrounding matrix
    size_type rowIndex() const
    {
      return row_;
    }

    //! the diagonal value
    const K& diagonal() const
    {
      return *p_;
    }

  protected:
    // the data, very simply a pointer to the diagonal value and the row number
    K* p_;
    size_type row_;

    void operator & ();
  };

  template< class K, int n >
  class DiagonalRowVector : public DiagonalRowVectorConst<K,n>
  {
  public:
    // standard constructor and everything is sufficient ...

    //===== type definitions and constants

    //! export the type representing the field
    typedef K field_type;

    //! export the type representing the components
    typedef K block_type;

    //! The type used for the index access and size operation
    typedef std::size_t size_type;

    //! Constructor making uninitialized vector
    DiagonalRowVector() : DiagonalRowVectorConst<K,n>()
    {}

    //! Constructor making vector with identical coordinates
    explicit DiagonalRowVector (K* p, int col) : DiagonalRowVectorConst<K,n>(p, col)
    {}

    //===== assignment from scalar
    //! Assignment operator for scalar
    DiagonalRowVector& operator= (const K& k)
    {
      *p_ = k;
      return *this;
    }

    //===== access to components

    //! random access
    K& operator[] (size_type i)
    {
      if (i!=row_)
        DUNE_THROW(FMatrixError,"index is read only");
      //            return ZERO<K>::Value;
      return *p_;
    }

    //! Iterator class for sequential access
    typedef ReferenceStorageIterator<DiagonalRowVector<K,n>,K,K&,DiagonalRowVector<K,n> > Iterator;
    //! typedef for stl compliant access
    typedef Iterator iterator;

    //! begin iterator
    Iterator begin ()
    {
      return Iterator(*this,row_);
    }

    //! end iterator
    Iterator end ()
    {
      return Iterator(*this,row_+1);
    }

    //! begin iterator
    Iterator rbegin ()
    {
      return Iterator(*this,row_);
    }

    //! end iterator
    Iterator rend ()
    {
      return Iterator(*this,row_-1);
    }

    //! return iterator to given element or end()
    //    Iterator find (size_type i)
    //    {
    //        if (i<n)
    //            return Iterator(*this,i);
    //        else
    //            return Iterator(*this,n);
    //    }

    //! ConstIterator class for sequential access
    typedef ReferenceStorageIterator<DiagonalRowVectorConst<K,n>,const K,const K&,DiagonalRowVectorConst<K,n> > ConstIterator;
    //! typedef for stl compliant access
    typedef ConstIterator const_iterator;

    using DiagonalRowVectorConst<K,n>::identical;
    using DiagonalRowVectorConst<K,n>::operator[];
    using DiagonalRowVectorConst<K,n>::operator==;
    using DiagonalRowVectorConst<K,n>::begin;
    using DiagonalRowVectorConst<K,n>::end;
    using DiagonalRowVectorConst<K,n>::rbegin;
    using DiagonalRowVectorConst<K,n>::rend;
    using DiagonalRowVectorConst<K,n>::N;
    using DiagonalRowVectorConst<K,n>::dim;
    using DiagonalRowVectorConst<K,n>::rowIndex;
    using DiagonalRowVectorConst<K,n>::diagonal;

  private:
    using DiagonalRowVectorConst<K,n>::p_;
    using DiagonalRowVectorConst<K,n>::row_;

    void operator & ();
  };


  // implement type traits
  template<class K, int n>
  struct const_reference< DiagonalRowVector<K,n> >
  {
    typedef DiagonalRowVectorConst<K,n> type;
  };

  template<class K, int n>
  struct const_reference< DiagonalRowVectorConst<K,n> >
  {
    typedef DiagonalRowVectorConst<K,n> type;
  };

  template<class K, int n>
  struct mutable_reference< DiagonalRowVector<K,n> >
  {
    typedef DiagonalRowVector<K,n> type;
  };

  template<class K, int n>
  struct mutable_reference< DiagonalRowVectorConst<K,n> >
  {
    typedef DiagonalRowVector<K,n> type;
  };


  template<class C, class T, class R, class S>
  class ReferenceStorageIterator : public BidirectionalIteratorFacade<ReferenceStorageIterator<C,T,R,S>,T, R, int>
  {
    friend class ReferenceStorageIterator<typename mutable_reference<C>::type, typename mutable_reference<T>::type, typename mutable_reference<R>::type, typename mutable_reference<S>::type>;
    friend class ReferenceStorageIterator<typename const_reference<C>::type, typename const_reference<T>::type, typename const_reference<R>::type, typename const_reference<S>::type>;

    typedef ReferenceStorageIterator<typename mutable_reference<C>::type, typename mutable_reference<T>::type, typename mutable_reference<R>::type, typename mutable_reference<S>::type> MyType;
    typedef ReferenceStorageIterator<typename const_reference<C>::type, typename const_reference<T>::type, typename const_reference<R>::type, typename const_reference<S>::type> MyConstType;

  public:

    // Constructors needed by the facade iterators.
    ReferenceStorageIterator() : container_(0), position_(0)
    {}

    ReferenceStorageIterator(C& cont, int pos)
      : container_(cont), position_(pos)
    {}

    ReferenceStorageIterator(const MyType& other)
      : container_(other.container_), position_(other.position_)
    {}

    ReferenceStorageIterator(const MyConstType& other)
      : container_(other.container_), position_(other.position_)
    {}

    // Methods needed by the forward iterator
    bool equals(const MyType& other) const
    {
      // check for identity since we store references/objects
      return position_ == other.position_ && container_.identical(other.container_);
    }


    bool equals(const MyConstType& other) const
    {
      // check for identity since we store references/objects
      return position_ == other.position_ && container_.identical(other.container_);
    }

    R dereference() const
    {
      // iterator facedes cast to const
      // thus this costness must be removed since the 'const'
      // container is '[...]Const' and not 'const [...]'
      return (const_cast< typename remove_const<C>::type& >(container_))[position_];
    }

    void increment()
    {
      ++position_;
    }

    // Additional function needed by BidirectionalIterator
    void decrement()
    {
      --position_;
    }

    // Additional function needed by RandomAccessIterator
    R elementAt(int i) const
    {
      return (const_cast< typename remove_const<C>::type& >(container_))[position_+i];
    }

    void advance(int n)
    {
      position_=position_+n;
    }

    std::ptrdiff_t distanceTo(const MyType& other) const
    {
      assert(container_.identical(other.container_));
      return other.position_ - position_;
    }

    std::ptrdiff_t distanceTo(const MyConstType& other) const
    {
      assert(container_.identical(other.container_));
      return other.position_ - position_;
    }

    std::ptrdiff_t index() const
    {
      return position_;
    }

  private:
    S container_;
    size_t position_;
  };



  template<class K, int n>
  void istl_assign_to_fmatrix(FieldMatrix<K,n,n>& fm, const DiagonalMatrix<K,n>& s)
  {
    fm = K();
    for(int i=0; i<n; ++i)
      fm[i][i] = s.diagonal()[i];
  }

} // end namespace
#endif
