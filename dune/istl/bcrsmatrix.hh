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
#include <map>

#include "istlexception.hh"
#include "bvector.hh"
#include "matrixutils.hh"
#include <dune/common/shared_ptr.hh>
#include <dune/common/stdstreams.hh>
#include <dune/common/iteratorfacades.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/ftraits.hh>
#include <dune/common/static_assert.hh>

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

  template<typename M>
  struct MatrixDimension;

  /**
   * \brief export statistic about compression achieved in mymode mode
   *
   * To enable the user to tune parameters of mymode buildmode of a
   * Dune::BCRSMatrix manually, some statistics are exported upon
   * compression.
   *
   */
  template<typename size_type>
  struct CompressionStatistics
  {
    //! average number of non-zeroes per row
    double avg;
    //! maximum number of non-zeroes in a row_type
    size_type maximum;
    //! total number of elements written to the overflow area
    size_type overflow_total;
    //! percentage of wasted memory resulting from non-used overflow area
    double mem_ratio;
  };

  /**
     \brief A sparse block matrix with compressed row storage

     Implements a block compressed row storage scheme. The block
     type B can be any type implementing the matrix interface.

     Different ways to build up a compressed row
     storage matrix are supported:

     1. Row-wise scheme
     2. Random scheme
     3. mymode scheme

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

     \code
     #include<dune/common/fmatrix.hh>
     #include<dune/istl/bcrsmatrix.hh>

     ...

     typedef FieldMatrix<double,2,2> M;
     // third parameter is an optional upper bound for the number
     // of nonzeros. If given the matrix will use one array for all values
     // as opposed to one for each row.
     BCRSMatrix<M> B(4,4,12,BCRSMatrix<M>::row_wise);

     typedef BCRSMatrix<M>::CreateIterator Iter;

     for(Iter row=B.createbegin(); row!=B.createend(); ++row){
       // Add nonzeros for left neighbour, diagonal and right neighbour
       if(row.index()>0)
         row.insert(row.index()-1);
       row.insert(row.index());
       if(row.index()<B.N()-1)
         row.insert(row.index()+1);
     }

     // Now the sparsity pattern is fully set up and we can add values

     B[0][0]=2;
     ...
     \endcode

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

     3. mymode scheme
     With the above Random Scheme, the sparsity pattern often has to be determined
     and stored before the matrix is assembled. This leads to an increase of
     needs in memory and computation time. Often, one has good a priori
     knowledge about the number of entries a row contains on average. mymode
     mode tries to make use of that knowledge by allocating memory based on
     that average. Entries in rows with more non-zeroes are written to an overflow
     area. After all indices are added a compression procedure is used.
     To use this mode use the following methods:

     Construct the matrix via
      - BCRSMatrix(size_type _n, size_type _m, size_type _avg, double _overflowsize, BuildMode bm)
      - void setSize(size_type rows, size_type columns, size_type nnz=0) after setting
        the buildmode to mymode and the compression parameter via setMymodeParameters(size_type _avg, double _overflow)

     Start filling your matrix with entry(size_type row, size_type col).
     Full access is possible also during buildstage, although not as fast
     as after buildstage via the operator[] due to the necessity of searching
     the overflow area for some matrix elements.

     After the entry-method has been called for each matrix at least once,
     a call to compress() reorganizes the data into one array for further use.
     This sets the matrixs state to built. No matrix entries may be added after
     this step. The return type Dune::CompressionStatistics allows for some parameter optimization.

     Use of copy constructor, assignment operator and matrix vector arithmetics
     are not supported until the matrix is fully built.

     In the following sample code, an array with 28 entries will be reserved
     \code
     #include<dune/common/fmatrix.hh>
     #include<dune/istl/bcrsmatrix.hh>

     typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > M;
     M m(10, 10, 2, 0.4, M::mymode);

     //fill in some arbitrary entries, even operations on these would be possible,
     //you get a reference to the entry! the order of these statements is irrelevant!
     m.entry(0,0) = 0.;
     m.entry(8,0) = 0.;
     m.entry(1,8) = 0.; m.entry(1,0) = 0.; m.entry(1,5) = 0.;
     m.entry(2,0) = 0.;
     m.entry(3,5) = 0.; m.entry(3,0) = 0.;  m.entry(3,8) = 0.;
     m.entry(4,0) = 0.;
     m.entry(9,0) = 0.; m.entry(9,5) = 0.; m.entry(9,8) = 0.;
     m.entry(5,0) = 0.; m.entry(5,5) = 0.; m.entry(5,8) = 0.;
     m.entry(6,0) = 0.;
     m.entry(7,0) = 0.; m.entry(7,5) = 0.; m.entry(7,8) = 0.;



     // internally the index array now looks like this (second row are the row pointers):
     // xxxxxxxx0x800x500x050x050x05
     // ........|.|.|.|.|.|.|.|.|.|.
     // and the overflow area contains (1,5,0.0), (3,8,0.0), (5,8,0.0), (7,8,0.0), (9,8,0.0)
     // the data array has similar structure.

     //finish building by compressing the array
     Dune::CompressionStatistics<M::size_type> stats = m.compress();

     // internally the index array now looks like this:
     // 00580058005800580058xxxxxxxx
     // ||..||..||..||..||..........

     \endcode
   */
  template<class B, class A=std::allocator<B> >
  class BCRSMatrix
  {
    friend struct MatrixDimension<BCRSMatrix>;
  public:
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
       * @brief Build entries randomly with an educated guess on entries per row_type
       *
       * Allows random order generation as in random mode, but rowsizes do not need
       * to be given first. Instead an average number of non-zeroes per row is passed
       * to the constructor. Matrix setup is finished with compress(), full data access
       * during buidstage is possible.
       */
      mymode,
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

    //! @returns an iterator that is positioned before
    //! the end iterator of the rows, i.e. at the last row.
    Iterator beforeEnd ()
    {
      return Iterator(r,n-1);
    }

    //! @returns an iterator that is positioned before
    //! the first row of the matrix.
    Iterator beforeBegin ()
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

    //! @returns an iterator that is positioned before
    //! the end iterator of the rows. i.e. at the last row.
    ConstIterator beforeEnd() const
    {
      return ConstIterator(r,n-1);
    }

    //! @returns an iterator that is positioned before
    //! the first row of the matrix.
    ConstIterator beforeBegin () const
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
        r(0), a(0)
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

    //! \brief construct matrix with a known average number of entries per row
    /**
     * Constructs a matrix in mymode buildmode.
     *
     * @param _n number of rows of the matrix
     * @param _m number of columns of the matrix
     * @param _avg expected average number of entries per row
     * @param _overflowsize fraction of _n*_avg which is expected to be
     *   needed for elements that exceed _avg entries per row.
     *
     */
    BCRSMatrix (size_type _n, size_type _m, size_type _avg, double _overflowsize, BuildMode bm)
      : build_mode(bm), ready(notbuilt), avg(_avg), overflowsize(_overflowsize)
    {
      mymode_allocate(_n,_m);
    }

    /**
     * @brief copy constructor
     *
     * Does a deep copy as expected.
     */
    BCRSMatrix (const BCRSMatrix& Mat)
      : n(Mat.n), nnz(0)
    {
      // deep copy in global array
      size_type _nnz = Mat.nnz;

      // in case of row-wise allocation
      if (_nnz<=0)
      {
        _nnz = 0;
        for (size_type i=0; i<n; i++)
          _nnz += Mat.r[i].getsize();
      }

      j = Mat.j; // enable column index sharing, release array in case of row-wise allocation
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
     * defaults to 0). Must be omitted in mymode mode.
     */
    void setSize(size_type rows, size_type columns, size_type nnz=0)
    {
      // deallocate already setup memory
      deallocate();

      if (build_mode == mymode)
      {
        if (nnz>0)
          DUNE_THROW(Dune::ISTLError,"number of non-zeroes may not be set in mymode mode, use setMymodeParameters() instead");

        // mymode allocates differently
        mymode_allocate(rows,columns);
      }
      else
      {
        // allocate matrix memory
        allocate(rows, columns, nnz);
      }
    }

    /** @brief Set parameters needed for creation in mymode.
     *
     * Use this method before setSize() to define storage behaviour of a matrix
     * in mymode mode
     * @param _avg expected average number of entries per row
     * @param _overflowsize fraction of _n*_avg which is expected to be
     *   needed for elements that exceed _avg entries per row.
     */
    void setmMymodeParameters(size_type _avg, double _overflow)
    {
      avg = _avg;
      overflow = _overflow;
    }

    /**
     * @brief assignment
     *
     * Frees and reallocates space.
     * Both sparsity pattern and values are set from Mat.
     */
    BCRSMatrix& operator= (const BCRSMatrix& Mat)
    {
      // return immediately when self-assignment
      if (&Mat==this) return *this;

      // make it simple: ALWAYS throw away memory for a and j
      deallocate(false);

      // reallocate the rows if required
      if (n>0 && n!=Mat.n) {
        // free rows
        for(row_type *riter=r+(n-1), *rend=r-1; riter!=rend; --riter)
          rowAllocator_.destroy(riter);
        rowAllocator_.deallocate(r,n);
      }

      nnz=Mat.nnz;
      if (nnz<=0)
      {
        for (size_type i=0; i<Mat.n; i++)
          nnz += Mat.r[i].getsize();
      }

      // allocate a, share j
      j = Mat.j;
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
        : Mat(_Mat), i(_i), nnz(0), current_row(Mat.a, Mat.j.get(), 0)
      {
        if (i==0 && Mat.ready)
          DUNE_THROW(ISTLError,"creation only allowed for uninitialized matrix");
        if(Mat.build_mode!=row_wise)
        {
          if(Mat.build_mode==unknown)
            Mat.build_mode=row_wise;
          else
            DUNE_THROW(ISTLError,"creation only allowed if row wise allocation was requested in the constructor");
        }
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
            B*   a = Mat.allocator_.allocate(s);
            // use placement new to call constructor that allocates
            // additional memory.
            new (a) B[s];
            size_type* j = Mat.sizeAllocator_.allocate(s);
            Mat.r[i].set(s,a,j);
          }
        }else
          // setup empty row
          Mat.r[i].set(0,0,0);

        // initialize the j array for row i from pattern
        size_type k=0;
        size_type *j =  Mat.r[i].getindexptr();
        for (typename PatternType::const_iterator it=pattern.begin(); it!=pattern.end(); ++it)
          j[k++] = *it;

        // now go to next row
        i++;
        pattern.clear();

        // check if this was last row
        if (i==Mat.n)
        {
          Mat.ready = built;
          if(Mat.nnz>0)
          {
            // Set nnz to the exact number of nonzero blocks inserted
            // as some methods rely on it
            Mat.nnz=nnz;
            Mat.allocationSize = nnz;
          }
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
      size_type size() const
      {
        return pattern.size();
      }

    private:
      BCRSMatrix& Mat;     // the matrix we are defining
      size_type i;               // current row to be defined
      size_type nnz;             // count total number of nonzeros
      typedef std::set<size_type,std::less<size_type> > PatternType;
      PatternType pattern;     // used to compile entries in a row
      row_type current_row;     // row pointing to the current row to setup
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
        total += r[i].getsize();
      }

      if(nnz==0)
        // allocate/check memory
        allocate(n,m,total,false);
      else if(nnz<total)
        DUNE_THROW(ISTLError,"Specified number of nonzeros ("<<nnz<<") not "
                                                             <<"sufficient for calculated nonzeros ("<<total<<"! ");

      // set the window pointers correctly
      setWindowPointers(begin());

      // initialize j array with m (an invalid column index)
      // this indicates an unused entry
      for (size_type k=0; k<nnz; k++)
        j.get()[k] = m;
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

    //===== mymode creation interface

    //! \brief get reference to entry (row,col) of the matrix
    /*!
     * This method can only be used when the matrix is in mymode
     * building mode.
     *
     * A reference to entry (row, col) of the matrix is returned.
     * If this is the first call for (row, col) the matrix element
     * is created.
     *
     * This method can only be used while building the matrix,
     * after compression operator[] gives a much better performance.
     */
    B& entry(size_type row, size_type col)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (build_mode!=mymode)
        DUNE_THROW(ISTLError,"requires mymode build mode");
      if (ready==built)
        DUNE_THROW(ISTLError,"matrix already built up, use operator[] for entry access now");

      if (row >= n)
        DUNE_THROW(ISTLError,"row index exceeds matrix size");
      if (col >= m)
        DUNE_THROW(ISTLError,"column index exceeds matrix size");
#endif

      size_type* begin = r[row].getindexptr();
      size_type* end = begin + r[row].getsize();

      size_type* pos = std::find(begin, end, col);

      //treat the case that there was a match in the array
      if (pos != end)
        if (*pos == col)
        {
          std::ptrdiff_t offset = pos - r[row].getindexptr();
          B* aptr = r[row].getptr() + offset;

          return *aptr;
        }

      //determine whether overflow has to be taken into account or not
      if (r[row].getsize() == avg)
        return overflow[std::make_pair(row,col)];
      else
      {
        //modify index array
        *end = col;

        //do simulatenous operations on data array a
        std::ptrdiff_t offset = end - r[row].getindexptr();
        B* apos = r[row].getptr() + offset;

        //increase rowsize
        r[row].setsize(r[row].getsize()+1);

        //return reference to the newly created entry
        return *apos;
      }
    }

    //! @brief finishes the buildstage in mymode mode
    /*! once called, matrix is going to built state and no indices
     *  can be added to a matrix that is built in mymode mode.
     *
     *  performs compression of index and data arrays with linear
     *  complexity
     *
     *  An object with some statistics about the compression for
     *  future optimization is returned.
     */
    CompressionStatistics<size_type> compress()
    {
      if (build_mode!=mymode)
        DUNE_THROW(ISTLError,"requires mymode build mode");
      if (ready==built)
        DUNE_THROW(ISTLError,"matrix already built up, no more need for compression");

      //calculate statistics
      CompressionStatistics<size_type> stats;
      stats.overflow_total = overflow.size();

      //get insertion iterators pointing to one before start (for later use of ++it)
      size_type* jiit = j.get()-1;
      B* aiit = a-1;

      //get iterator to the smallest overflow element
      typename OverflowType::iterator oit = overflow.begin();

      //store a copy of index pointers on which to perform sortation
      std::vector<size_type*> perm;

      //iterate over all rows and copy elements into their position in the compressed array
      for (size_type i=0; i<n; i++)
      {
        //get old pointers into a and j and size without overflow changes
        size_type* begin = r[i].getindexptr();
        //B* apos = r[i].getptr();
        size_type size = r[i].getsize();

        perm.resize(size);

        typename std::vector<size_type*>::iterator it = perm.begin();
        for (size_type* iit = begin; iit < begin + size; ++iit, ++it)
          *it = iit;

        //sort permutation array
        std::sort(perm.begin(),perm.end(),PointerCompare<size_type>());

        //change row window pointer to their new positions
        r[i].setindexptr(jiit+1);
        r[i].setptr(aiit+1);

        for (it = perm.begin(); it != perm.end(); ++it)
        {
          //check whether there are elements in the overflow area which take precedence
          while ((oit!=overflow.end()) && (oit->first < std::make_pair(i,**it)))
          {
            //check whether there is enough memory to write to
            if (jiit >= begin)
              DUNE_THROW(Dune::ISTLError,"Allocated Size for BCRSMatrix was not sufficient!");
            //copy and element from the overflow area to the insertion position in a and j
            *(++jiit) = oit->first.second;
            *(++aiit) = oit->second;
            ++oit;
            r[i].setsize(r[i].getsize()+1);
          }

          //check whether there is enough memory to write to
          if (jiit >= begin)
            DUNE_THROW(Dune::ISTLError,"Allocated Size for BCRSMatrix was not sufficient!");

          //copy element from array
          *(++jiit) = **it;
          B* apos = *it-j.get()+a;
          *(++aiit) = *apos;
        }

        //copy remaining elements from the overflow area
        while ((oit!=overflow.end()) && (oit->first.first == i))
        {
          //check whether there is enough memory to write to
          if (jiit >= begin)
            DUNE_THROW(Dune::ISTLError,"Allocated Size for BCRSMatrix was not sufficient!");

          //copy and element from the overflow area to the insertion position in a and j
          *(++jiit) = oit->first.second;
          *(++aiit) = oit->second;
          ++oit;
          r[i].setsize(r[i].getsize()+1);
        }
      }

      // overflow area may be cleared
      overflow.clear();

      //determine average number of entries and memory usage
      std::ptrdiff_t diff = (r[n-1].getindexptr() + r[n-1].getsize() - j.get());
      nnz = diff;
      stats.avg = (double) (nnz) / (double) n;
      stats.mem_ratio = (double) (nnz)/(double) allocationSize;

      //determine maximum number of entries in a row
      stats.maximum = 0;
      for (size_type i=0; i<n; i++)
        if (r[i].getsize()>stats.maximum)
          stats.maximum = r[i].getsize();

      //matrix is now built
      ready = built;

      return stats;
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

    /*! \brief Add the scaled entries of another matrix to this one.
     *
     * Matrix axpy operation: *this += alpha * b
     *
     * \param alpha Scaling factor.
     * \param b     The matrix to add to this one. Its sparsity pattern has to
     *              be subset of the sparsity pattern of this matrix.
     */
    BCRSMatrix& axpy(field_type alpha, const BCRSMatrix& b)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if(N()!=b.N() || M() != b.M())
        DUNE_THROW(RangeError, "Matrix sizes do not match!");
#endif
      RowIterator endi=end();
      ConstRowIterator j=b.begin();
      for(RowIterator i=begin(); i!=endi; ++i, ++j)
        i->axpy(alpha, *j);

      return *this;
    }

    //===== linear maps

    //! y = A x
    template<class X, class Y>
    void mv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=M()) DUNE_THROW(ISTLError,
                                 "Size mismatch: M: " << N() << "x" << M() << " x: " << x.N());
      if (y.N()!=N()) DUNE_THROW(ISTLError,
                                 "Size mismatch: M: " << N() << "x" << M() << " y: " << y.N());
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
    typename FieldTraits<field_type>::real_type frobenius_norm2 () const
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
    typename FieldTraits<field_type>::real_type frobenius_norm () const
    {
      return sqrt(frobenius_norm2());
    }

    //! infinity norm (row sum norm, how to generalize for blocks?)
    typename FieldTraits<field_type>::real_type infinity_norm () const
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
    typename FieldTraits<field_type>::real_type infinity_norm_real () const
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

    //! number of blocks that are stored (the number of blocks that possibly are nonzero)
    size_type nonzeroes () const
    {
      return nnz;
    }

    BuildStage buildStage() const
    {
      return ready;
    }

    //===== query

    //! return true if (i,j) is in pattern
    bool exists (size_type i, size_type j) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (i<0 || i>=n) DUNE_THROW(ISTLError,"row index out of range");
      if (j<0 || j>=m) DUNE_THROW(ISTLError,"column index out of range");
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

    // The allocator used for memory management
    typename A::template rebind<B>::other allocator_;

    typename A::template rebind<row_type>::other rowAllocator_;

    typename A::template rebind<size_type>::other sizeAllocator_;

    // size of the matrix
    size_type n;       // number of rows
    size_type m;       // number of columns
    size_type nnz;     // number of nonzeroes contained in the matrix
    size_type allocationSize; //allocated size of a and j arrays, except in mymode mode: nnz==allocationsSize
    // zero means that memory is allocated separately for each row.

    // the rows are dynamically allocated
    row_type* r;     // [n] the individual rows having pointers into a,j arrays

    // dynamically allocated memory
    B*   a;      // [allocationSize] non-zero entries of the matrix in row-wise ordering
    // If a single array of column indices is used, it can be shared
    // between different matrices with the same sparsity pattern
    Dune::shared_ptr<size_type> j;  // [allocationSize] column indices of entries

    // additional data is needed in mymode buildmode
    size_type avg;
    double overflowsize;

    typedef std::map<std::pair<size_type,size_type>, B> OverflowType;
    OverflowType overflow;

    void setWindowPointers(ConstRowIterator row)
    {
      row_type current_row(a,j.get(),0); // Pointers to current row data
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

      if (allocationSize>0)
      {
        // a,j have been allocated as one long vector
        j.reset();
        for(B *aiter=a+(allocationSize-1), *aend=a-1; aiter!=aend; --aiter)
          allocator_.destroy(aiter);
        allocator_.deallocate(a,allocationSize);
      }
      else
      {
        // check if memory for rows have been allocated individually
        for (size_type i=0; i<n; i++)
          if (r[i].getsize()>0)
          {
            for (B *col=r[i].getptr()+(r[i].getsize()-1),
                 *colend = r[i].getptr()-1; col!=colend; --col) {
              allocator_.destroy(col);
            }
            sizeAllocator_.deallocate(r[i].getindexptr(),1);
            allocator_.deallocate(r[i].getptr(),1);
          }
      }

      // deallocate the rows
      if (n>0 && deallocateRows) {
        for(row_type *riter=r+(n-1), *rend=r-1; riter!=rend; --riter)
          rowAllocator_.destroy(riter);
        rowAllocator_.deallocate(r,n);
      }

      // Mark matrix as not built at all.
      ready=notbuilt;

    }

    /** \brief Class used by shared_ptr to deallocate memory using the proper allocator */
    class Deallocator
    {
      typename A::template rebind<size_type>::other& sizeAllocator_;

    public:
      Deallocator(typename A::template rebind<size_type>::other& sizeAllocator)
        : sizeAllocator_(sizeAllocator)
      {}

      void operator()(size_type* p) { sizeAllocator_.deallocate(p,1); }
    };


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
     * @param allocationSize_ The number of nonzero entries the matrix should hold (if omitted
     * defaults to 0).
     * @param allocateRow Whether we have to allocate the row pointers, too. (Defaults to
     * true)
     */
    void allocate(size_type rows, size_type columns, size_type allocationSize_=0, bool allocateRows=true)
    {
      // Store size
      n = rows;
      m = columns;
      nnz = allocationSize_;
      allocationSize = allocationSize_;

      // allocate rows
      if(allocateRows) {
        if (n>0) {
          r = rowAllocator_.allocate(rows);
        }else{
          r = 0;
        }
      }

      // allocate a and j array
      if (allocationSize>0) {
        a = allocator_.allocate(allocationSize);
        // use placement new to call constructor that allocates
        // additional memory.
        new (a) B[allocationSize];

        // allocate column indices only if not yet present (enable sharing)
        if (!j.get())
          j.reset(sizeAllocator_.allocate(allocationSize),Deallocator(sizeAllocator_));
      }else{
        a = 0;
        j.reset();
        for(row_type* ri=r; ri!=r+rows; ++ri)
          rowAllocator_.construct(ri, row_type());
      }

      // Mark the matrix as not built.
      ready = notbuilt;
    }

    /** @brief organizes allocation mymode mode
     * calculates correct array size to be allocated and sets the
     * the window pointers to their correct positions for insertion.
     * internally uses allocate() for the real allocation.
     */
    void mymode_allocate(size_type _n, size_type _m)
    {
      //calculate size of overflow region, add buffer for row sort!
      size_type osize = (size_type) (_n*avg)*overflowsize + 4*avg;
      allocationSize = _n*avg + osize;

      allocate(_n, _m, allocationSize);

      //set row pointers correctly
      size_type* jptr = j.get() + osize;
      B* aptr = a + osize;
      for (size_type i=0; i<n; i++)
      {
        r[i].set(0,aptr,jptr);
        jptr = jptr + avg;
        aptr = aptr + avg;
      }
    }
  };


  /** @} end documentation */

} // end namespace

#endif
