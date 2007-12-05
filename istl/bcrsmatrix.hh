// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_BCRSMATRIX_HH
#define DUNE_BCRSMATRIX_HH

#include <cmath>
#include <complex>
#include <set>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

#include "istlexception.hh"
#include "allocator.hh"
#include "bvector.hh"
#include <dune/common/stdstreams.hh>
#include <dune/common/iteratorfacades.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/helpertemplates.hh>

/*! \file
 * \brief Implementation of the BCRSMatrix class
 */

namespace Dune {

  /**
   * @defgroup ISTL_SPMV Sparse Matrix and Vector classes
   * @ingroup ISTL
   * @brief Matrix and Vector classes that support a block recursive
   * structure capable of representing the natural structure from Finite
   * Element discretisations.
   *
   *
   * The interface of our matrices is designed according to what they
   * represent from a mathematical point of view. The vector classes are
   * representations of vector spaces:
   *
   * - FieldVector represents a vector space \f$V=K^n\f$ where the field \f$K\f$
   *   is represented by a numeric type (e.g. double, float, complex). \f$n\f$
   *   is known at compile time.
   * - BlockVector represents a vector space \f$V=W\times W \times W \times\cdots\times W\f$
   *   where W is itself a vector space.
   * - VariableBlockVector represents a vector space having a two-level
   *   block structure of the form
   *   \f$V=B^{n_1}\times B^{n_2}\times\ldots \times B^{n_m}\f$, i.e. it is constructed
   *   from \f$m\f$ vector spaces, \f$i=1,\ldots,m\f$.
   *
   * The matrix classes represent linear maps \f$A: V \mapsto W\f$
   * from vector space \f$V\f$ to vector space \f$W\f$ the recursive block
   * structure of the matrix rows and columns immediately follows
   * from the recursive block structure of the vectors representing
   * the domain and range of the mapping, respectively:
   * - FieldMatrix represents a linear map \f$M: V_1 \to V_2\f$ where
   *   \f$V_1=K^n\f$ and \f$V_2=K^m\f$ are vector spaces over the same field represented by a numerix type.
   * - BCRSMatrix represents a linear map \f$M: V_1 \to V_2\f$ where
   *   \f$V_1=W\times W \times W \times\cdots\times W\f$ and \f$V_2=W\times W \times W \times\cdots\times W\f$
   *   where W is itself a vector space.
   * - VariableBCRSMatrix is not yet implemented.
   */
  /**
              @addtogroup ISTL_SPMV
              @{
   */


  /**
     \brief A sparse block matrix with compressed row storage

     Implements a block compressed row storage scheme. The block
     type B can be any type implementing the matrix interface.

     Different ways to build up a compressed row
     storage matrix are supported:

     1. Row-wise scheme
     2. Random scheme

     Error checking: no error checking is provided normally.
     Setting the compile time switch DUNE_ISTL_WITH_CHECKING
     enables error checking.

     Details:

     1. Row-wise scheme

     Rows are built up in sequential order. Size of the row and
     the column indices are defined. A row can be used as soon as it
     is initialized. With respect to memory there are two variants of
     this scheme: (a) number of non-zeroes known in advance (application
     finite difference schemes), (b) number of non-zeroes not known
     in advance (application: Sparse LU, ILU(n)).

     2. Random scheme

     For general finite element implementations the number of rows n
     is known, the number of non-zeroes might also be known (e.g.
     \#edges + \#nodes for P1) but the size of a row and the indices of a row
     can not be defined in sequential order.

     \code
     #include<dune/common/fmatrix.hh>
     #include<dune/istl/bcrsmatrix.hh>

     ...

     typedef FieldMatrix<double,2,2> M;
     BCRSMatrix<M> B(4,4,BCRSMatrix<M>::random);

     // initially set row size for each row
     B.setrowsize(0,1);
     B.setrowsize(3,4);
     B.setrowsize(2,1);
     B.setrowsize(1,1);
     // increase row size for row 2
     B.incrementrowsize(2)

     // finalize row setup phase
     B.endrowsizes();

     // add column entries to rows
     B.addindex(0,0);
     B.addindex(3,1);
     B.addindex(2,2);
     B.addindex(1,1);
     B.addindex(2,0);
     B.addindex(3,2);
     B.addindex(3,0);
     B.addindex(3,3);

     // finalize column setup phase
     B.endindices();

     // set entries using the random access operator
     B[0][0] = 1;
     B[1][1] = 2;
     B[2][0] = 3;
     B[2][2] = 4;
     B[3][1] = 5;
     B[3][2] = 6;
     B[3][0] = 7;
     B[3][3] = 8;
     \endcode
   */
#ifdef DUNE_EXPRESSIONTEMPLATES
  template<class B, class A>
  class BCRSMatrix : public ExprTmpl::Matrix< BCRSMatrix<B,A> >
#else
  template<class B, class A=ISTLAllocator>
  class BCRSMatrix
#endif
  {
  private:
    enum BuildStage {
      /** @brief Matrix is not built at all. */
      notbuilt=0,
      /** @brief The row sizes of the matrix are known.
       *
       * Only used in random mode.
       */
      rowSizesBuilt=1,
      /** @brief The matrix structure is built fully.*/
      built=2
    };

  public:

    //===== type definitions and constants

    //! export the type representing the field
    typedef typename B::field_type field_type;

    //! export the type representing the components
    typedef B block_type;

    //! export the allocator type
    typedef A allocator_type;

    //! implement row_type with compressed vector
    typedef CompressedBlockVectorWindow<B,A> row_type;

    //! The type for the index access and the size
    typedef typename A::size_type size_type;

    //! increment block level counter
    enum {
      //! The number of blocklevels the matrix contains.
      blocklevel = B::blocklevel+1
    };

    //! we support two modes
    enum BuildMode {
      /**
       * @brief Build in a row-wise manner.
       *
       * Rows are built up in sequential order. Size of the row and
       * the column indices are defined. A row can be used as soon as it
       * is initialized. With respect to memory there are two variants of
       * this scheme: (a) number of non-zeroes known in advance (application
       * finite difference schemes), (b) number of non-zeroes not known
       * in advance (application: Sparse LU, ILU(n)).
       */
      row_wise,
      /**
       * @brief Build entries randomly.
       *
       * For general finite element implementations the number of rows n
       * is known, the number of non-zeroes might also be known (e.g.
       * \#edges + \#nodes for P1) but the size of a row and the indices of a row
       * can not be defined in sequential order.
       */
      random,
      /**
       * @brief Build mode not set!
       */
      unknown
    };


    //===== random access interface to rows of the matrix

    //! random access to the rows
    row_type& operator[] (size_type i)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (r==0) DUNE_THROW(ISTLError,"row not initialized yet");
      if (i>=n) DUNE_THROW(ISTLError,"index out of range");
      if (r[i].getptr()==0) DUNE_THROW(ISTLError,"row not initialized yet");
#endif
      return r[i];
    }

    //! same for read only access
    const row_type& operator[] (size_type i) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (built!=ready) DUNE_THROW(ISTLError,"row not initialized yet");
      if (i>=n) DUNE_THROW(ISTLError,"index out of range");
#endif
      return r[i];
    }


    //===== iterator interface to rows of the matrix

    //! %Iterator access to matrix rows
    template<class T>
    class RealRowIterator
      : public RandomAccessIteratorFacade<RealRowIterator<T>, T>
    {

    public:
      //! \brief The unqualified value type
      typedef typename remove_const<T>::type ValueType;

      friend class RandomAccessIteratorFacade<RealRowIterator<const ValueType>, const ValueType>;
      friend class RandomAccessIteratorFacade<RealRowIterator<ValueType>, ValueType>;
      friend class RealRowIterator<const ValueType>;
      friend class RealRowIterator<ValueType>;

      //! constructor
      RealRowIterator (row_type* _p, size_type _i)
        : p(_p), i(_i)
      {}

      //! empty constructor, use with care!
      RealRowIterator ()
        : p(0), i(0)
      {}

      RealRowIterator(const RealRowIterator<ValueType>& it)
        : p(it.p), i(it.i)
      {}


      //! return index
      size_type index () const
      {
        return i;
      }

      std::ptrdiff_t distanceTo(const RealRowIterator<ValueType>& other) const
      {
        assert(other.p==p);
        return (other.i-i);
      }

      std::ptrdiff_t distanceTo(const RealRowIterator<const ValueType>& other) const
      {
        assert(other.p==p);
        return (other.i-i);
      }

      //! equality
      bool equals (const RealRowIterator<ValueType>& other) const
      {
        assert(other.p==p);
        return i==other.i;
      }

      //! equality
      bool equals (const RealRowIterator<const ValueType>& other) const
      {
        assert(other.p==p);
        return i==other.i;
      }

    private:
      //! prefix increment
      void increment()
      {
        ++i;
      }

      //! prefix decrement
      void decrement()
      {
        --i;
      }

      void advance(std::ptrdiff_t diff)
      {
        i+=diff;
      }

      T& elementAt(std::ptrdiff_t diff) const
      {
        return p[i+diff];
      }

      //! dereferencing
      row_type& dereference () const
      {
        return p[i];
      }

      row_type* p;
      size_type i;
    };

    //! The iterator over the (mutable matrix rows
    typedef RealRowIterator<row_type> iterator;
    typedef RealRowIterator<row_type> Iterator;


    //! Get iterator to first row
    Iterator begin ()
    {
      return Iterator(r,0);
    }

    //! Get iterator to one beyond last row
    Iterator end ()
    {
      return Iterator(r,n);
    }

    //! Get iterator to last row
    Iterator rbegin ()
    {
      return Iterator(r,n-1);
    }

    //! Get iterator to one before first row
    Iterator rend ()
    {
      return Iterator(r,-1);
    }

    //! rename the iterators for easier access
    typedef Iterator RowIterator;

    /** \brief Iterator for the entries of each row */
    typedef typename row_type::Iterator ColIterator;

    //! The const iterator over the matrix rows
    typedef RealRowIterator<const row_type> const_iterator;
    typedef RealRowIterator<const row_type> ConstIterator;


    //! Get const iterator to first row
    ConstIterator begin () const
    {
      return ConstIterator(r,0);
    }

    //! Get const iterator to one beyond last row
    ConstIterator end () const
    {
      return ConstIterator(r,n);
    }

    //! Get const iterator to last row
    ConstIterator rbegin () const
    {
      return ConstIterator(r,n-1);
    }

    //! Get const iterator to one before first row
    ConstIterator rend () const
    {
      return ConstIterator(r,-1);
    }

    //! rename the const row iterator for easier access
    typedef ConstIterator ConstRowIterator;

    //! Const iterator to the entries of a row
    typedef typename row_type::ConstIterator ConstColIterator;

    //===== constructors & resizers

    //! an empty matrix
    BCRSMatrix ()
      : build_mode(unknown), ready(notbuilt), n(0), m(0), nnz(0),
        r(0), a(0), j(0)
    {}

    //! matrix with known number of nonzeroes
    BCRSMatrix (size_type _n, size_type _m, size_type _nnz, BuildMode bm)
      : build_mode(bm), ready(notbuilt)
    {
      allocate(_n, _m, _nnz);
    }

    //! matrix with unknown number of nonzeroes
    BCRSMatrix (size_type _n, size_type _m, BuildMode bm)
      : build_mode(bm), ready(notbuilt)
    {
      allocate(_n, _m);
    }

    //! copy constructor
    BCRSMatrix (const BCRSMatrix& Mat)
      : n(Mat.n), nnz(0)
    {
      // deep copy in global array
      int _nnz = Mat.nnz;

      // in case of row-wise allocation
      if (_nnz<=0)
      {
        _nnz = 0;
        for (size_type i=0; i<n; i++)
          _nnz += Mat.r[i].getsize();
      }

      allocate(Mat.n, Mat.m, _nnz);

      // build window structure
      copyWindowStructure(Mat);
    }

    //! destructor
    ~BCRSMatrix ()
    {
      deallocate();
    }

    /**
     * @brief Sets the build mode of the matrix
     * @param bm The build mode to use.
     */
    void setBuildMode(BuildMode bm)
    {
      if(ready==notbuilt)
        build_mode = bm;
      else
        DUNE_THROW(InvalidStateException, "Matrix structure is already built (ready="<<ready<<").");
    }

    /**
     *  @brief Set the size of the matrix.
     *
     * Sets the number of rows and columns of the matrix and allocates
     * the memory needed for the storage of the matrix entries.
     *
     * @warning After calling this methods on an already allocated (and probably
     * setup matrix) results in all the structure and data being deleted. I.~e.
     * one has to setup the matrix again.
     *
     * @param rows The number of rows the matrix should contain.
     * @param columns the number of columns the matrix should contain.
     * @param nnz The number of nonzero entries the matrix should hold (if omitted
     * defaults to 0).
     */
    void setSize(size_type rows, size_type columns, size_type nnz=0)
    {
      // deallocate already setup memory
      deallocate();

      // allocate matrix memory
      allocate(rows, columns, nnz);
    }

    //! assignment
    BCRSMatrix& operator= (const BCRSMatrix& Mat)
    {
      // return immediately when self-assignment
      if (&Mat==this) return *this;

      // make it simple: ALWAYS throw away memory for a and j
      deallocate(false);

      // reallocate the rows if required
      if (n>0 && n!=Mat.n)
        // free rows
        A::template free<row_type>(r);

      nnz=Mat.nnz;
      if (nnz<=0)
      {
        for (size_type i=0; i<Mat.n; i++)
          nnz += Mat.r[i].getsize();
      }

      // allocate a,j
      allocate(Mat.n, Mat.m, nnz, n!=Mat.n);

      // build window structure
      copyWindowStructure(Mat);
      return *this;
    }

    //! Assignment from a scalar
    BCRSMatrix& operator= (const field_type& k)
    {
      for (size_type i=0; i<n; i++) r[i] = k;
      return *this;
    }

    //===== row-wise creation interface

    //! %Iterator class for sequential creation of blocks
    class CreateIterator
    {
    public:
      //! constructor
      CreateIterator (BCRSMatrix& _Mat, size_type _i)
        : Mat(_Mat), i(_i), nnz(0), current_row(Mat.a, Mat.j, 0)
      {
        if (i==0 && Mat.ready)
          DUNE_THROW(ISTLError,"creation only allowed for uninitialized matrix");
        if(Mat.build_mode!=row_wise)
          if(Mat.build_mode==unknown)
            Mat.build_mode=row_wise;
          else
            DUNE_THROW(ISTLError,"creation only allowed if row wise allocation was requested in the constructor");
      }

      //! prefix increment
      CreateIterator& operator++()
      {
        // this should only be called if matrix is in creation
        if (Mat.ready)
          DUNE_THROW(ISTLError,"matrix already built up");

        // row i is defined through the pattern
        // get memory for the row and initialize the j array
        // this depends on the allocation mode

        // compute size of the row
        size_type s = pattern.size();

        if(s>0) {
          // update number of nonzeroes including this row
          nnz += s;

          // alloc memory / set window
          if (Mat.nnz>0)
          {
            // memory is allocated in one long array

            // check if that memory is sufficient
            if (nnz>Mat.nnz)
              DUNE_THROW(ISTLError,"allocated nnz too small");

            // set row i
            Mat.r[i].set(s,current_row.getptr(),current_row.getindexptr());
            current_row.setptr(current_row.getptr()+s);
            current_row.setindexptr(current_row.getindexptr()+s);
          }else{
            // memory is allocated individually per row
            // allocate and set row i
            B*   a = A::template malloc<B>(s);
            size_type* j = A::template malloc<size_type>(s);
            Mat.r[i].set(s,a,j);
          }
        }else
          // setup empty row
          Mat.r[i].set(0,0,0);

        // initialize the j array for row i from pattern
        size_type k=0;
        size_type *j =  Mat.r[i].getindexptr();
        for (typename std::set<size_type>::const_iterator it=pattern.begin(); it!=pattern.end(); ++it)
          j[k++] = *it;

        // now go to next row
        i++;
        pattern.clear();

        // check if this was last row
        if (i==Mat.n)
        {
          Mat.ready = built;
          if(Mat.nnz>0)
            // Set nnz to the exact number of nonzero blocks inserted
            // as some methods rely on it
            Mat.nnz=nnz;
        }
        // done
        return *this;
      }

      //! inequality
      bool operator!= (const CreateIterator& it) const
      {
        return (i!=it.i) || (&Mat!=&it.Mat);
      }

      //! equality
      bool operator== (const CreateIterator& it) const
      {
        return (i==it.i) && (&Mat==&it.Mat);
      }

      //! dereferencing
      size_type index () const
      {
        return i;
      }

      //! put column index in row
      void insert (size_type j)
      {
        pattern.insert(j);
      }

      //! return true if column index is in row
      bool contains (size_type j)
      {
        if (pattern.find(j)!=pattern.end())
          return true;
        else
          return false;
      }
      /**
       * @brief Get the current row size.
       * @return The number of indices already
       * inserted for the current row.
       */
      typename std::set<size_type>::size_type size() const
      {
        return pattern.size();
      }

    private:
      BCRSMatrix& Mat;     // the matrix we are defining
      size_type i;               // current row to be defined
      size_type nnz;             // count total number of nonzeros
      std::set<size_type> pattern;     // used to compile entries in a row
      row_type current_row;     // row poiting to the current row to setup
    };

    //! allow CreateIterator to access internal data
    friend class CreateIterator;

    //! get initial create iterator
    CreateIterator createbegin ()
    {
      return CreateIterator(*this,0);
    }

    //! get create iterator pointing to one after the last block
    CreateIterator createend ()
    {
      return CreateIterator(*this,n);
    }


    //===== random creation interface

    //! set number of indices in row i to s
    void setrowsize (size_type i, size_type s)
    {
      if (build_mode!=random)
        DUNE_THROW(ISTLError,"requires random build mode");
      if (ready)
        DUNE_THROW(ISTLError,"matrix row sizes already built up");

      r[i].setsize(s);
    }

    //! get current number of indices in row i
    size_type getrowsize (size_type i) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (r==0) DUNE_THROW(ISTLError,"row not initialized yet");
      if (i>=n) DUNE_THROW(ISTLError,"index out of range");
#endif
      return r[i].getsize();
    }

    //! increment size of row i by s (1 by default)
    void incrementrowsize (size_type i, size_type s = 1)
    {
      if (build_mode!=random)
        DUNE_THROW(ISTLError,"requires random build mode");
      if (ready)
        DUNE_THROW(ISTLError,"matrix row sizes already built up");

      r[i].setsize(r[i].getsize()+s);
    }

    //! indicate that size of all rows is defined
    void endrowsizes ()
    {
      if (build_mode!=random)
        DUNE_THROW(ISTLError,"requires random build mode");
      if (ready)
        DUNE_THROW(ISTLError,"matrix row sizes already built up");

      // compute total size, check positivity
      size_type total=0;
      for (size_type i=0; i<n; i++)
      {
        if (r[i].getsize()<0)
          DUNE_THROW(ISTLError,"rowsize must be nonnegative");
        total += r[i].getsize();
      }

      // allocate/check memory
      allocate(n,m,total,false);

      // set the window pointers correctly
      setWindowPointers(begin());

      // initialize j array with m (an invalid column index)
      // this indicates an unused entry
      for (size_type k=0; k<nnz; k++)
        j[k] = m;
      ready = rowSizesBuilt;
    }

    //! \brief add index (row,col) to the matrix
    /*!
       This method can only be used when building the BCRSMatrix
       in random mode.

       addindex adds a new column entry to the row. If this column
       entry already exists, nothing is done.

       Don't call addindex after the setup phase is finished
       (after endindices is called).
     */
    void addindex (size_type row, size_type col)
    {
      if (build_mode!=random)
        DUNE_THROW(ISTLError,"requires random build mode");
      if (ready==built)
        DUNE_THROW(ISTLError,"matrix already built up");
      if (ready==notbuilt)
        DUNE_THROW(ISTLError,"matrix row sizes not built up yet");

      if (col >= m)
        DUNE_THROW(ISTLError,"column index exceeds matrix size");

      // get row range
      size_type* const first = r[row].getindexptr();
      size_type* const last = first + r[row].getsize();

      // find correct insertion position for new column index
      size_type* pos = std::lower_bound(first,last,col);

      // check if index is already in row
      if (pos!=last && *pos == col) return;

      // find end of already inserted column indices
      size_type* end = std::lower_bound(pos,last,m);
      if (end==last)
        DUNE_THROW(ISTLError,"row is too small");

      // insert new column index at correct position
      std::copy_backward(pos,end,end+1);
      *pos = col;

    }

    //! indicate that all indices are defined, check consistency
    void endindices ()
    {
      if (build_mode!=random)
        DUNE_THROW(ISTLError,"requires random build mode");
      if (ready==built)
        DUNE_THROW(ISTLError,"matrix already built up");
      if (ready==notbuilt)
        DUNE_THROW(ISTLError,"row sizes are not built up yet");

      // check if there are undefined indices
      RowIterator endi=end();
      for (RowIterator i=begin(); i!=endi; ++i)
      {
        ColIterator endj = (*i).end();
        for (ColIterator j=(*i).begin(); j!=endj; ++j) {
          if (j.index()<0)
          {
            std::cout << "j[" << j.offset() << "]=" << j.index() << std::endl;
            DUNE_THROW(ISTLError,"undefined index detected");
          }
          if (j.index()>=m) {
            dwarn << "WARNING: size of row "<< i.index()<<" is "<<j.offset()<<". But was specified as being "<< (*i).end().offset()
                  <<". This means you are wasting valuable space and creating additional cache misses!"<<std::endl;
            r[i.index()].setsize(j.offset());
            break;
          }
        }
      }

      // if not, set matrix to built
      ready = built;
    }

    //===== vector space arithmetic

    //! vector space multiplication with scalar
    BCRSMatrix& operator*= (const field_type& k)
    {
      if (nnz>0)
      {
        // process 1D array
        for (size_type i=0; i<nnz; i++)
          a[i] *= k;
      }
      else
      {
        RowIterator endi=end();
        for (RowIterator i=begin(); i!=endi; ++i)
        {
          ColIterator endj = (*i).end();
          for (ColIterator j=(*i).begin(); j!=endj; ++j)
            (*j) *= k;
        }
      }

      return *this;
    }

    //! vector space division by scalar
    BCRSMatrix& operator/= (const field_type& k)
    {
      if (nnz>0)
      {
        // process 1D array
        for (size_type i=0; i<nnz; i++)
          a[i] /= k;
      }
      else
      {
        RowIterator endi=end();
        for (RowIterator i=begin(); i!=endi; ++i)
        {
          ColIterator endj = (*i).end();
          for (ColIterator j=(*i).begin(); j!=endj; ++j)
            (*j) /= k;
        }
      }

      return *this;
    }


    /*! \brief Add the entries of another matrix to this one.
     *
     * \param b The matrix to add to this one. Its sparsity pattern
     * has to be subset of the sparsity pattern of this matrix.
     */
    BCRSMatrix& operator+= (const BCRSMatrix& b)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if(N()!=b.N() || M() != b.M())
        DUNE_THROW(RangeError, "Matrix sizes do not match!");
#endif
      RowIterator endi=end();
      ConstRowIterator j=b.begin();
      for (RowIterator i=begin(); i!=endi; ++i, ++j) {
        i->operator+=(*j);
      }

      return *this;
    }

    /*! \brief Substract the entries of another matrix to this one.
     *
     * \param b The matrix to add to this one. Its sparsity pattern
     * has to be subset of the sparsity pattern of this matrix.
     */
    BCRSMatrix& operator-= (const BCRSMatrix& b)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if(N()!=b.N() || M() != b.M())
        DUNE_THROW(RangeError, "Matrix sizes do not match!");
#endif
      RowIterator endi=end();
      ConstRowIterator j=b.begin();
      for (RowIterator i=begin(); i!=endi; ++i, ++j) {
        i->operator-=(*j);
      }

      return *this;
    }
    //===== linear maps

    //! y = A x
    template<class X, class Y>
    void mv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        y[i.index()]=0;
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          (*j).umv(x[j.index()],y[i.index()]);
      }
    }

    //! y += A x
    template<class X, class Y>
    void umv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          (*j).umv(x[j.index()],y[i.index()]);
      }
    }

    //! y -= A x
    template<class X, class Y>
    void mmv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          (*j).mmv(x[j.index()],y[i.index()]);
      }
    }

    //! y += alpha A x
    template<class X, class Y>
    void usmv (const field_type& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          (*j).usmv(alpha,x[j.index()],y[i.index()]);
      }
    }

    //! y += A^T x
    template<class X, class Y>
    void umtv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          (*j).umtv(x[i.index()],y[j.index()]);
      }
    }

    //! y -= A^T x
    template<class X, class Y>
    void mmtv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          (*j).mmtv(x[i.index()],y[j.index()]);
      }
    }

    //! y += alpha A^T x
    template<class X, class Y>
    void usmtv (const field_type& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          (*j).usmtv(alpha,x[i.index()],y[j.index()]);
      }
    }

    //! y += A^H x
    template<class X, class Y>
    void umhv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          (*j).umhv(x[i.index()],y[j.index()]);
      }
    }

    //! y -= A^H x
    template<class X, class Y>
    void mmhv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          (*j).mmhv(x[i.index()],y[j.index()]);
      }
    }

    //! y += alpha A^H x
    template<class X, class Y>
    void usmhv (const field_type& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(ISTLError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(ISTLError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          (*j).usmhv(alpha,x[i.index()],y[j.index()]);
      }
    }


    //===== norms

    //! square of frobenius norm, need for block recursion
    double frobenius_norm2 () const
    {
      double sum=0;

      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          sum += (*j).frobenius_norm2();
      }

      return sum;
    }

    //! frobenius norm: sqrt(sum over squared values of entries)
    double frobenius_norm () const
    {
      return sqrt(frobenius_norm2());
    }

    //! infinity norm (row sum norm, how to generalize for blocks?)
    double infinity_norm () const
    {
      double max=0;
      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        double sum=0;
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          sum += (*j).infinity_norm();
        max = std::max(max,sum);
      }
      return max;
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    double infinity_norm_real () const
    {
      double max=0;
      ConstRowIterator endi=end();
      for (ConstRowIterator i=begin(); i!=endi; ++i)
      {
        double sum=0;
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
          sum += (*j).infinity_norm_real();
        max = std::max(max,sum);
      }
      return max;
    }


    //===== sizes

    //! number of rows (counted in blocks)
    size_type N () const
    {
      return n;
    }

    //! number of columns (counted in blocks)
    size_type M () const
    {
      return m;
    }

    /** \brief row dimension of block r
        \bug Does not count empty rows (FlySpray #7)
     */
    size_type rowdim (size_type i) const
    {
      const B* row = r[i].getptr();
      if(row)
        return row->rowdim();
      else
        return 0;
    }

    /** \brief Column dimension of block c
        \bug Does not count empty columns (FlySpray #7)
     */
    size_type coldim (size_type c) const
    {
      // find an entry in column j
      if (nnz>0)
      {
        for (size_type k=0; k<nnz; k++) {
          if (j[k]==c) {
            return a[k].coldim();
          }
        }
      }
      else
      {
        for (size_type i=0; i<n; i++)
        {
          size_type* j = r[i].getindexptr();
          B*   a = r[i].getptr();
          for (size_type k=0; k<r[i].getsize(); k++)
            if (j[k]==c) {
              return a[k].coldim();
            }
        }
      }

      // not found
      return 0;
    }

    //! dimension of the destination vector space
    size_type rowdim () const
    {
      size_type nn=0;
      for (size_type i=0; i<n; i++)
        nn += rowdim(i);
      return nn;
    }

    //! dimension of the source vector space
    size_type coldim () const
    {
      // The following code has a complexity of nnz, and
      // typically a very small constant.
      //
      std::vector<size_type> coldims(M(),-1);
      for (ConstRowIterator row=begin(); row!=end(); ++row)
        for (ConstColIterator col=row->begin(); col!=row->end(); ++col)
          // only compute blocksizes we don't already have
          if (coldims[col.index()]==-1)
            coldims[col.index()] = col->coldim();

      size_type sum = 0;
      for (typename std::vector<size_type>::iterator it=coldims.begin(); it!=coldims.end(); ++it)
        // skip rows for which no coldim could be determined
        if ((*it)>=0)
          sum += *it;

      return sum;
    }

    //===== query

    //! return true if (i,j) is in pattern
    bool exists (size_type i, size_type j) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (i<0 || i>=n) DUNE_THROW(ISTLError,"index out of range");
      if (j<0 || i>=m) DUNE_THROW(ISTLError,"index out of range");
#endif
      if (r[i].size() && r[i].find(j)!=r[i].end())
        return true;
      else
        return false;
    }


  private:
    // state information
    BuildMode build_mode;     // row wise or whole matrix
    BuildStage ready;               // indicate the stage the matrix building is in

    // size of the matrix
    size_type n;       // number of rows
    size_type m;       // number of columns
    size_type nnz;     // number of nonzeros allocated in the a and j array below
    // zero means that memory is allocated separately for each row.

    // the rows are dynamically allocated
    row_type* r;     // [n] the individual rows having pointers into a,j arrays

    // dynamically allocated memory
    B*   a;      // [nnz] non-zero entries of the matrix in row-wise ordering
    size_type* j;      // [nnz] column indices of entries

    void setWindowPointers(ConstRowIterator row)
    {
      row_type current_row(a,j,0); // Pointers to current row data
      for (size_type i=0; i<n; i++, ++row) {
        // set row i
        size_type s = row->getsize();

        if (s>0) {
          // setup pointers and size
          r[i].set(s,current_row.getptr(), current_row.getindexptr());
          // update pointer for next row
          current_row.setptr(current_row.getptr()+s);
          current_row.setindexptr(current_row.getindexptr()+s);
        } else{
          // empty row
          r[i].set(0,0,0);
        }
      }
    }

    //! \brief Copy the window structure from another matrix
    void copyWindowStructure(const BCRSMatrix& Mat)
    {
      setWindowPointers(Mat.begin());

      // copy data
      for (size_type i=0; i<n; i++) r[i] = Mat.r[i];

      // finish off
      build_mode = row_wise; // dummy
      ready = built;
    }

    /**
     * @brief deallocate memory of the matrix.
     * @param deallocateRows Whether we have to deallocate the row pointers, too.
     * If false they will not be touched. (Defaults to true).
     */
    void deallocate(bool deallocateRows=true)
    {

      if (nnz>0)
      {
        // a,j have been allocated as one long vector
        A::template free<size_type>(j);
        A::template free<B>(a);
      }
      else
      {
        // check if memory for rows have been allocated individually
        for (size_type i=0; i<n; i++)
          if (r[i].getsize()>0)
          {
            A::template free<size_type>(r[i].getindexptr());
            A::template free<B>(r[i].getptr());
          }
      }

      // deallocate the rows
      if (n>0 && deallocateRows) A::template free<row_type>(r);

      // Mark matrix as not built at all.
      ready=notbuilt;

    }

    /**
     *  @brief Allocate memory for the matrix structure
     *
     * Sets the number of rows and columns of the matrix and allocates
     * the memory needed for the storage of the matrix entries.
     *
     * @warning After calling this methods on an already allocated (and probably
     * setup matrix) results in all the structure and data being lost. Please
     * call deallocate() before calling allocate in this case.
     *
     * @param row The number of rows the matrix should contain.
     * @param columns the number of columns the matrix should contain.
     * @param nnz The number of nonzero entries the matrix should hold (if omitted
     * defaults to 0).
     * @param allocateRow Whether we have to allocate the row pointers, too. (Defaults to
     * true)
     */
    void allocate(size_type rows, size_type columns, size_type nnz_=0, bool allocateRows=true)
    {
      // Store size
      n = rows;
      m = columns;
      nnz = nnz_;

      // allocate rows
      if(allocateRows) {
        if (n>0) {
          r = A::template malloc<row_type>(rows);
        }else{
          r = 0;
        }
      }


      // allocate a and j array
      if (nnz>0) {
        a = A::template malloc<B>(nnz);
        j = A::template malloc<size_type>(nnz);
      }else{
        a = 0;
        j = 0;
      }
      // Mark the matrix as not built.
      ready = notbuilt;
    }

  };


#ifdef DUNE_EXPRESSIONTEMPLATES
  template <class B, class A>
  struct FieldType< BCRSMatrix<B,A> >
  {
    typedef typename FieldType<B>::type type;
  };

  template <class B, class A>
  struct BlockType< BCRSMatrix<B,A> >
  {
    typedef B type;
  };
  template <class B, class A>
  struct RowType< BCRSMatrix<B,A> >
  {
    typedef CompressedBlockVectorWindow<B,A> type;
  };
#endif

  /** @} end documentation */

} // end namespace

#endif
