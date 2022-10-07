// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_BCRSMATRIX_HH
#define DUNE_ISTL_BCRSMATRIX_HH

#include <cmath>
#include <complex>
#include <set>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <map>
#include <memory>

#include "istlexception.hh"
#include "bvector.hh"
#include "matrixutils.hh"
#include <dune/common/stdstreams.hh>
#include <dune/common/iteratorfacades.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/ftraits.hh>
#include <dune/common/scalarvectorview.hh>
#include <dune/common/scalarmatrixview.hh>

#include <dune/istl/blocklevel.hh>

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

  //! Statistics about compression achieved in implicit mode.
  /**
   * To enable the user to tune parameters of the implicit build mode of a
   * Dune::BCRSMatrix manually, some statistics are exported upon during
   * the compression step.
   *
   */
  template<typename size_type>
  struct CompressionStatistics
  {
    //! average number of non-zeroes per row.
    double avg;
    //! maximum number of non-zeroes per row.
    size_type maximum;
    //! total number of elements written to the overflow area during construction.
    size_type overflow_total;
    //! fraction of wasted memory resulting from non-used overflow area.
    /**
     * mem_ratio is equal to `nonzeros()/(# allocated matrix entries)`.
     */
    double mem_ratio;
  };

  //! A wrapper for uniform access to the BCRSMatrix during and after the build stage in implicit build mode.
  /**
   * The implicit build mode of Dune::BCRSMatrix handles matrices differently during
   * assembly and afterwards. Using this class, one can wrap a BCRSMatrix to allow
   * use with code that is not written for the new build mode specifically. The wrapper
   * forwards any calls to operator[][] to the entry() method.The assembly code
   * does not even necessarily need to know that the underlying matrix is sparse.
   * Dune::AMG uses this to reassemble an existing matrix without code duplication.
   * The compress() method of Dune::BCRSMatrix still has to be called from outside
   * this wrapper after the pattern assembly is finished.
   *
   * \tparam M_ the matrix type
   */
  template<class M_>
  class ImplicitMatrixBuilder
  {

  public:

    //! The underlying matrix.
    typedef M_ Matrix;

    //! The block_type of the underlying matrix.
    typedef typename Matrix::block_type block_type;

    //! The size_type of the underlying matrix.
    typedef typename Matrix::size_type size_type;

    //! Proxy row object for entry access.
    /**
     * During matrix construction, there are no fully functional rows available
     * yet, so we instead hand out a simple proxy which only allows accessing
     * individual entries using operator[].
     */
    class row_object
    {

    public:

      //! Returns entry in column j.
      block_type& operator[](size_type j) const
      {
        return _m.entry(_i,j);
      }

#ifndef DOXYGEN

      row_object(Matrix& m, size_type i)
        : _m(m)
        , _i(i)
      {}

#endif

    private:

      Matrix& _m;
      size_type _i;

    };

    //! Creates an ImplicitMatrixBuilder for matrix m.
    /**
     * \note You can only pass a completely set up matrix to this constructor:
     *       All of setBuildMode(), setImplicitBuildModeParameters() and setSize()
     *       must have been called with the correct values.
     *
     */
    ImplicitMatrixBuilder(Matrix& m)
      : _m(m)
    {
      if (m.buildMode() != Matrix::implicit)
        DUNE_THROW(BCRSMatrixError,"You can only create an ImplicitBuilder for a matrix in implicit build mode");
      if (m.buildStage() != Matrix::building)
        DUNE_THROW(BCRSMatrixError,"You can only create an ImplicitBuilder for a matrix with set size that has not been compressed() yet");
    }

    //! Sets up matrix m for implicit construction using the given parameters and creates an ImplicitBmatrixuilder for it.
    /**
     * Using this constructor, you can perform the necessary matrix setup and the creation
     * of the ImplicitMatrixBuilder in a single step. The matrix must still be in the build stage
     * notAllocated, otherwise a BCRSMatrixError will be thrown. For a detailed explanation
     * of the matrix parameters, see BCRSMatrix.
     *
     * \param m                 the matrix to be built
     * \param rows              the number of matrix rows
     * \param cols              the number of matrix columns
     * \param avg_cols_per_row  the average number of non-zero columns per row
     * \param overflow_fraction the amount of overflow to reserve in the matrix
     *
     * \sa BCRSMatrix
     */
    ImplicitMatrixBuilder(Matrix& m, size_type rows, size_type cols, size_type avg_cols_per_row, double overflow_fraction)
      : _m(m)
    {
      if (m.buildStage() != Matrix::notAllocated)
        DUNE_THROW(BCRSMatrixError,"You can only set up a matrix for this ImplicitBuilder if it has no memory allocated yet");
      m.setBuildMode(Matrix::implicit);
      m.setImplicitBuildModeParameters(avg_cols_per_row,overflow_fraction);
      m.setSize(rows,cols);
    }

    //! Returns a proxy for entries in row i.
    row_object operator[](size_type i) const
    {
      return row_object(_m,i);
    }

    //! The number of rows in the matrix.
    size_type N() const
    {
      return _m.N();
    }

    //! The number of columns in the matrix.
    size_type M() const
    {
      return _m.M();
    }

  private:

    Matrix& _m;

  };

  /**
     \brief A sparse block matrix with compressed row storage

     Implements a block compressed row storage scheme. The block
     type B can be any type implementing the matrix interface.

     Different ways to build up a compressed row
     storage matrix are supported:

     1. Row-wise scheme
     2. Random scheme
     3. implicit scheme

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
     \#edges + \#nodes for P2) but the size of a row and the indices of a row
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
     B.incrementrowsize(2);

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

     3. implicit scheme

     With the 'random scheme` described above, the sparsity pattern has to be determined
     and stored before the matrix is assembled. This requires a dedicated iteration
     over the grid elements, which can be costly in terms of time. Also, additional
     memory is needed to store the pattern before it can be given to the 'random' build mode.

     On the other hand, often one has good a priori knowledge about the number of entries
     a row contains on average. The `implicit` mode tries to make use of that knowledge,
     and allows the setup of matrix pattern and numerical values together.

     Constructing and filling a BCRSMatrix with the 'implicit' mode is performed in two steps:
     In a setup phase, matrix entries with
     numerical values can be inserted into the matrix. Then, a compression algorithm is called which defragments
     and optimizes the memory layout. After this compression step, the matrix is
     ready to be used, and no further nonzero entries can be added.

     To use this mode, either construct a matrix object via

      - BCRSMatrix(size_type _n, size_type _m, size_type _avg, double compressionBufferSize, BuildMode bm)

     or default-construct the matrix and then call
      - setImplicitBuildModeParameters(size_type _avg, double compressionBufferSize)
      - setSize(size_type rows, size_type columns, size_type nnz=0)

     The parameter `_avg` specifies the expected number of (block) entries per matrix row.

     When the BCRSMatrix object is first constructed with the 'implicit' build mode,
     two areas for matrix entry storage are allocated:

     1) A large continuous chunk of memory that can hold the expected number of entries.
        In addition, this chunk contains an extra part of memory called the 'compression buffer',
        located before the memory for the matrix itself.
        The size of this buffer will be `_avg * _n * compressionBufferSize`.

     2) An associative container indexed by \f$i,j\f$-pairs, which will hold surplus
        matrix entries during the setup phase (the 'overflow area'). Its content is merged into the main
        memory during compression.

     You can then start filling your matrix by calling entry(size_type row, size_type col),
     which returns the corresponding matrix entry, creating it on the fly if
     it does not exist yet.
     The matrix pattern is hence created implicitly by simply accessing nonzero entries
     during the initial matrix assembly.  Note that new entries are not zero-initialized,
     though, and hence the first operation on each entry has to be an assignment.

     If a row contains more non-zero entries than what was specified in the _avg parameter,
     the surplus entries are stored in the 'overflow area' during the initial setup phase.
     After all indices are added, call compress() to trigger the compression step
     that optimizes the matrix and
     integrates any entries from the overflow area into the standard BCRS storage.
     This compression step builds up the final memory layout row by row.
     It will fail with an exception if the compression buffer is not large enough, which would lead
     to compressed rows overwriting uncompressed ones.
     More precisely, if \f$\textrm{nnz}_j\f$ denotes the
     number of non-zeros in the \f$j\f$-th row, then the compression algorithm will succeed
     if the maximal number of non-zeros in the \f$i\f$-th row is
     \f[
        M_i = \textrm{avg} + A + \sum_{j<i} (\textrm{avg} - \textrm{nnz}_j)
     \f]
     for all \f$i\f$, where
     \f$ A = \textrm{avg}(n \cdot \textrm{compressionBufferSize}) \f$
     is the total size of the compression buffer determined by the parameters
     explained above.

     The data of the matrix is now located at the beginning of the allocated
     area, and covers what used to be the compression buffer. In exchange, there is now
     unused space at the end of the large allocated piece of memory. This will go
     unused and cannot be freed during the lifetime of the matrix, but it has no
     negative impact on run-time performance. No matrix entries may be added after
     the compression step.

     The compress() method returns a value of type Dune::CompressionStatistics, which
     you can inspect to tune the construction parameters `_avg` and `compressionBufferSize`.

     Use of copy constructor, assignment operator and matrix vector arithmetics
     are not supported until the matrix is fully built.

     The following sample code constructs a \f$ 10 \times 10\f$ matrix, with an expected
     number of two entries per matrix row.  The compression buffer size is set to 0.4.
     Hence the main chunk of allocated memory will be able to hold `10 * 2` entries
     in the matrix rows, and `10 * 2 * 0.4` entries in the compression buffer.
     In total that's 28 entries.
     \code
     #include<dune/istl/bcrsmatrix.hh>

     typedef Dune::BCRSMatrix<double> M;
     M m(10, 10, 2, 0.4, M::implicit);

     // Fill in some arbitrary entries; the order is irrelevant.
     // Even operations on these would be possible, you get a reference to the entry!
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
     \endcode

     Internally the index array now looks like this:
     \code
     // xxxxxxxx0x800x500x050x050x05
     // ........|.|.|.|.|.|.|.|.|.|.
     \endcode
     The second row denotes the beginnings of the matrix rows.
     The eight 'x' on the left are the compression buffer.
     The overflow area contains the entries (1,5,0.0), (3,8,0.0), (5,8,0.0), (7,8,0.0),
     and (9,8,0.0).
     These are entries of rows 1, 3, 5, 7, and 9, which have three entries each,
     even though only two were anticipated.

     \code
     //finish building by compressing the array
     Dune::CompressionStatistics<M::size_type> stats = m.compress();
     \endcode

     Internally the index array now looks like this:
     \code
     // 00580058005800580058xxxxxxxx
     // ||..||..||..||..||..........
     \endcode
     The compression buffer on the left is gone now, and the matrix has a real CRS layout.
     The 'x' on the right will be unused for the rest of the matrix' lifetime.
   */
  template<class B, class A=std::allocator<B> >
  class BCRSMatrix
  {
    friend struct MatrixDimension<BCRSMatrix>;
  public:
    enum BuildStage {
      /** @brief Matrix is not built at all, no memory has been allocated, build mode and size can still be set. */
      notbuilt=0,
      /** @brief Matrix is not built at all, no memory has been allocated, build mode and size can still be set. */
      notAllocated=0,
      /** @brief Matrix is currently being built, some memory has been allocated, build mode and size are fixed. */
      building=1,
      /** @brief The row sizes of the matrix are known.
       *
       * Only used in random mode.
       */
      rowSizesBuilt=2,
      /** @brief The matrix structure is fully built. */
      built=3
    };

    //===== type definitions and constants

    //! export the type representing the field
    using field_type = typename Imp::BlockTraits<B>::field_type;

    //! export the type representing the components
    typedef B block_type;

    //! export the allocator type
    typedef A allocator_type;

    //! implement row_type with compressed vector
    typedef Imp::CompressedBlockVectorWindow<B,A> row_type;

    //! The type for the index access and the size
    typedef typename A::size_type size_type;

    //! The type for the statistics object returned by compress()
    typedef ::Dune::CompressionStatistics<size_type> CompressionStatistics;

    //! increment block level counter
    [[deprecated("Use free function blockLevel(). Will be removed after 2.8.")]]
    static constexpr unsigned int blocklevel = blockLevel<B>()+1;

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
       * \#edges + \#nodes for P2) but the size of a row and the indices of a row
       * cannot be defined in sequential order.
       */
      random,
      /**
       * @brief Build entries randomly with an educated guess for the number of entries per row.
       *
       * Allows random order generation as in random mode, but row sizes do not need
       * to be given first. Instead an average number of non-zeroes per row is passed
       * to the constructor. Matrix setup is finished with compress(), full data access
       * during build stage is possible.
       */
      implicit,
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
      if (build_mode == implicit && ready != built)
        DUNE_THROW(BCRSMatrixError,"You cannot use operator[] in implicit build mode before calling compress()");
      if (r==0) DUNE_THROW(BCRSMatrixError,"row not initialized yet");
      if (i>=n) DUNE_THROW(BCRSMatrixError,"index out of range");
#endif
      return r[i];
    }

    //! same for read only access
    const row_type& operator[] (size_type i) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (build_mode == implicit && ready != built)
        DUNE_THROW(BCRSMatrixError,"You cannot use operator[] in implicit build mode before calling compress()");
      if (built!=ready) DUNE_THROW(BCRSMatrixError,"row not initialized yet");
      if (i>=n) DUNE_THROW(BCRSMatrixError,"index out of range");
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
      typedef typename std::remove_const<T>::type ValueType;

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

    // we use a negative compressionBufferSize to indicate that the implicit
    // mode parameters have not been set yet

    //! an empty matrix
    BCRSMatrix ()
      : build_mode(unknown), ready(notAllocated), n(0), m(0), nnz_(0),
        allocationSize_(0), r(0), a(0),
        avg(0), compressionBufferSize_(-1.0)
    {}

    //! matrix with known number of nonzeroes
    BCRSMatrix (size_type _n, size_type _m, size_type _nnz, BuildMode bm)
      : build_mode(bm), ready(notAllocated), n(0), m(0), nnz_(0),
        allocationSize_(0), r(0), a(0),
        avg(0), compressionBufferSize_(-1.0)
    {
      allocate(_n, _m, _nnz,true,false);
    }

    //! matrix with unknown number of nonzeroes
    BCRSMatrix (size_type _n, size_type _m, BuildMode bm)
      : build_mode(bm), ready(notAllocated), n(0), m(0), nnz_(0),
        allocationSize_(0), r(0), a(0),
        avg(0), compressionBufferSize_(-1.0)
    {
      allocate(_n, _m,0,true,false);
    }

    //! \brief construct matrix with a known average number of entries per row
    /**
     * Constructs a matrix in implicit buildmode.
     *
     * @param _n number of rows of the matrix
     * @param _m number of columns of the matrix
     * @param _avg expected average number of entries per row
     * @param compressionBufferSize fraction of _n*_avg which is expected to be
     *   needed for elements that exceed _avg entries per row.
     *
     */
    BCRSMatrix (size_type _n, size_type _m, size_type _avg, double compressionBufferSize, BuildMode bm)
      : build_mode(bm), ready(notAllocated), n(0), m(0), nnz_(0),
        allocationSize_(0), r(0), a(0),
        avg(_avg), compressionBufferSize_(compressionBufferSize)
    {
      if (bm != implicit)
        DUNE_THROW(BCRSMatrixError,"Only call this constructor when using the implicit build mode");
      // Prevent user from setting a negative compression buffer size:
      // 1) It doesn't make sense
      // 2) We use a negative value to indicate that the parameters
      //    have not been set yet
      if (compressionBufferSize_ < 0.0)
        DUNE_THROW(BCRSMatrixError,"You cannot set a negative overflow fraction");
      implicit_allocate(_n,_m);
    }

    /**
     * @brief copy constructor
     *
     * Does a deep copy as expected.
     */
    BCRSMatrix (const BCRSMatrix& Mat)
      : build_mode(Mat.build_mode), ready(notAllocated), n(0), m(0), nnz_(0),
        allocationSize_(0), r(0), a(0),
        avg(Mat.avg), compressionBufferSize_(Mat.compressionBufferSize_)
    {
      if (!(Mat.ready == notAllocated || Mat.ready == built))
        DUNE_THROW(InvalidStateException,"BCRSMatrix can only be copy-constructed when source matrix is completely empty (size not set) or fully built)");

      // deep copy in global array
      size_type _nnz = Mat.nonzeroes();

      j_ = Mat.j_; // enable column index sharing, release array in case of row-wise allocation
      allocate(Mat.n, Mat.m, _nnz, true, true);

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
      if (ready == notAllocated)
        {
          build_mode = bm;
          return;
        }
      if (ready == building && (build_mode == unknown || build_mode == random || build_mode == row_wise) && (bm == row_wise || bm == random))
        build_mode = bm;
      else
        DUNE_THROW(InvalidStateException, "Matrix structure cannot be changed at this stage anymore (ready == "<<ready<<").");
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
     * defaults to 0). Must be omitted in implicit mode.
     */
    void setSize(size_type rows, size_type columns, size_type nnz=0)
    {
      // deallocate already setup memory
      deallocate();

      if (build_mode == implicit)
      {
        if (nnz>0)
          DUNE_THROW(Dune::BCRSMatrixError,"number of non-zeroes may not be set in implicit mode, use setImplicitBuildModeParameters() instead");

        // implicit allocates differently
        implicit_allocate(rows,columns);
      }
      else
      {
        // allocate matrix memory
        allocate(rows, columns, nnz, true, false);
      }
    }

    /** @brief Set parameters needed for creation in implicit build mode.
     *
     * Use this method before setSize() to define storage behaviour of a matrix
     * in implicit build mode
     * @param _avg expected average number of entries per row
     * @param compressionBufferSize fraction of _n*_avg which is expected to be
     *   needed for elements that exceed _avg entries per row.
     */
    void setImplicitBuildModeParameters(size_type _avg, double compressionBufferSize)
    {
      // Prevent user from setting a negative compression buffer size:
      // 1) It doesn't make sense
      // 2) We use a negative value to indicate that the parameters
      //    have not been set yet
      if (compressionBufferSize < 0.0)
        DUNE_THROW(BCRSMatrixError,"You cannot set a negative compressionBufferSize value");

      // make sure the parameters aren't changed after memory has been allocated
      if (ready != notAllocated)
        DUNE_THROW(InvalidStateException,"You cannot modify build mode parameters at this stage anymore");
      avg = _avg;
      compressionBufferSize_ = compressionBufferSize;
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

      if (!((ready == notAllocated || ready == built) && (Mat.ready == notAllocated || Mat.ready == built)))
        DUNE_THROW(InvalidStateException,"BCRSMatrix can only be copied when both target and source are empty or fully built)");

      // make it simple: ALWAYS throw away memory for a and j_
      // and deallocate rows only if n != Mat.n
      deallocate(n!=Mat.n);

      // reallocate the rows if required
      if (n>0 && n!=Mat.n) {
        // free rows
        for(row_type *riter=r+(n-1), *rend=r-1; riter!=rend; --riter)
          std::allocator_traits<decltype(rowAllocator_)>::destroy(rowAllocator_, riter);
        rowAllocator_.deallocate(r,n);
      }

      nnz_ = Mat.nonzeroes();

      // allocate a, share j_
      j_ = Mat.j_;
      allocate(Mat.n, Mat.m, nnz_, n!=Mat.n, true);

      // build window structure
      copyWindowStructure(Mat);
      return *this;
    }

    //! Assignment from a scalar
    BCRSMatrix& operator= (const field_type& k)
    {

      if (!(ready == notAllocated || ready == built))
        DUNE_THROW(InvalidStateException,"Scalar assignment only works on fully built BCRSMatrix)");

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
        : Mat(_Mat), i(_i), nnz(0), current_row(nullptr, Mat.j_.get(), 0)
      {
        if (Mat.build_mode == unknown && Mat.ready == building)
          {
            Mat.build_mode = row_wise;
          }
        if (i==0 && Mat.ready != building)
          DUNE_THROW(BCRSMatrixError,"creation only allowed for uninitialized matrix");
        if(Mat.build_mode!=row_wise)
          DUNE_THROW(BCRSMatrixError,"creation only allowed if row wise allocation was requested in the constructor");
        if(i==0 && _Mat.N()==0)
          // empty Matrix is always built.
           Mat.ready = built;
      }

      //! prefix increment
      CreateIterator& operator++()
      {
        // this should only be called if matrix is in creation
        if (Mat.ready != building)
          DUNE_THROW(BCRSMatrixError,"matrix already built up");

        // row i is defined through the pattern
        // get memory for the row and initialize the j_ array
        // this depends on the allocation mode

        // compute size of the row
        size_type s = pattern.size();

        if(s>0) {
          // update number of nonzeroes including this row
          nnz += s;

          // alloc memory / set window
          if (Mat.nnz_ > 0)
          {
            // memory is allocated in one long array

            // check if that memory is sufficient
            if (nnz > Mat.nnz_)
              DUNE_THROW(BCRSMatrixError,"allocated nnz too small");

            // set row i
            Mat.r[i].set(s,nullptr,current_row.getindexptr());
            current_row.setindexptr(current_row.getindexptr()+s);
          }else{
            // memory is allocated individually per row
            // allocate and set row i
            B* b = Mat.allocator_.allocate(s);
            // use placement new to call constructor that allocates
            // additional memory.
            new (b) B[s];
            size_type* j = Mat.sizeAllocator_.allocate(s);
            Mat.r[i].set(s,b,j);
          }
        }else
          // setup empty row
          Mat.r[i].set(0,nullptr,nullptr);

        // initialize the j array for row i from pattern
        std::copy(pattern.cbegin(), pattern.cend(), Mat.r[i].getindexptr());

        // now go to next row
        i++;
        pattern.clear();

        // check if this was last row
        if (i==Mat.n)
        {
          Mat.ready = built;
          if(Mat.nnz_ > 0)
          {
            // Set nnz to the exact number of nonzero blocks inserted
            // as some methods rely on it
            Mat.nnz_ = nnz;
            // allocate data array
            Mat.allocateData();
            Mat.setDataPointers();
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

      //! The number of the row that the iterator currently points to
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
        return pattern.find(j) != pattern.end();
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

    /** \brief Set number of indices in row i to s
     *
     * The number s may actually be larger than the true number of nonzero entries in row i.  In that
     * case, the extra memory goes wasted.  You will receive run-time warnings about this, sent to
     * the Dune::dwarn stream.
     */
    void setrowsize (size_type i, size_type s)
    {
      if (build_mode!=random)
        DUNE_THROW(BCRSMatrixError,"requires random build mode");
      if (ready != building)
        DUNE_THROW(BCRSMatrixError,"matrix row sizes already built up");

      r[i].setsize(s);
    }

    //! get current number of indices in row i
    size_type getrowsize (size_type i) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (r==0) DUNE_THROW(BCRSMatrixError,"row not initialized yet");
      if (i>=n) DUNE_THROW(BCRSMatrixError,"index out of range");
#endif
      return r[i].getsize();
    }

    //! increment size of row i by s (1 by default)
    void incrementrowsize (size_type i, size_type s = 1)
    {
      if (build_mode!=random)
        DUNE_THROW(BCRSMatrixError,"requires random build mode");
      if (ready != building)
        DUNE_THROW(BCRSMatrixError,"matrix row sizes already built up");

      r[i].setsize(r[i].getsize()+s);
    }

    //! indicate that size of all rows is defined
    void endrowsizes ()
    {
      if (build_mode!=random)
        DUNE_THROW(BCRSMatrixError,"requires random build mode");
      if (ready != building)
        DUNE_THROW(BCRSMatrixError,"matrix row sizes already built up");

      // compute total size, check positivity
      size_type total=0;
      for (size_type i=0; i<n; i++)
      {
        total += r[i].getsize();
      }

      if(nnz_ == 0)
        // allocate/check memory
        allocate(n,m,total,false,false);
      else if(nnz_ < total)
        DUNE_THROW(BCRSMatrixError,"Specified number of nonzeros ("<<nnz_<<") not "
                                                             <<"sufficient for calculated nonzeros ("<<total<<"! ");

      // set the window pointers correctly
      setColumnPointers(begin());

      // initialize j_ array with m (an invalid column index)
      // this indicates an unused entry
      for (size_type k=0; k<nnz_; k++)
        j_.get()[k] = m;
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
        DUNE_THROW(BCRSMatrixError,"requires random build mode");
      if (ready==built)
        DUNE_THROW(BCRSMatrixError,"matrix already built up");
      if (ready==building)
        DUNE_THROW(BCRSMatrixError,"matrix row sizes not built up yet");
      if (ready==notAllocated)
        DUNE_THROW(BCRSMatrixError,"matrix size not set and no memory allocated yet");

      if (col >= m)
        DUNE_THROW(BCRSMatrixError,"column index exceeds matrix size");

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
        DUNE_THROW(BCRSMatrixError,"row is too small");

      // insert new column index at correct position
      std::copy_backward(pos,end,end+1);
      *pos = col;
    }

    //! Set all column indices for row from the given iterator range.
    /**
     * The iterator range has to be of the same length as the previously set row size.
     * The entries in the iterator range do not have to be in any particular order, but
     * must not contain duplicate values.
     *
     * Calling this method overwrites any previously set column indices!
     */
    template<typename It>
    void setIndices(size_type row, It begin, It end)
    {
      size_type row_size = r[row].size();
      size_type* col_begin = r[row].getindexptr();
      size_type* col_end;
      // consistency check between allocated row size and number of passed column indices
      if ((col_end = std::copy(begin,end,r[row].getindexptr())) != col_begin + row_size)
        DUNE_THROW(BCRSMatrixError,"Given size of row " << row
                   << " (" << row_size
                   << ") does not match number of passed entries (" << (col_end - col_begin) << ")");
      std::sort(col_begin,col_end);
    }

    //! indicate that all indices are defined, check consistency
    void endindices ()
    {
      if (build_mode!=random)
        DUNE_THROW(BCRSMatrixError,"requires random build mode");
      if (ready==built)
        DUNE_THROW(BCRSMatrixError,"matrix already built up");
      if (ready==building)
        DUNE_THROW(BCRSMatrixError,"row sizes are not built up yet");
      if (ready==notAllocated)
        DUNE_THROW(BCRSMatrixError,"matrix size not set and no memory allocated yet");

      // check if there are undefined indices
      RowIterator endi=end();
      for (RowIterator i=begin(); i!=endi; ++i)
      {
        ColIterator endj = (*i).end();
        for (ColIterator j=(*i).begin(); j!=endj; ++j) {
          if (j.index() >= m) {
            dwarn << "WARNING: size of row "<< i.index()<<" is "<<j.offset()<<". But was specified as being "<< (*i).end().offset()
                  <<". This means you are wasting valuable space and creating additional cache misses!"<<std::endl;
            nnz_ -= ((*i).end().offset() - j.offset());
            r[i.index()].setsize(j.offset());
            break;
          }
        }
      }

      allocateData();
      setDataPointers();

      // if not, set matrix to built
      ready = built;
    }

    //===== implicit creation interface

    //! Returns reference to entry (row,col) of the matrix.
    /**
     * This method can only be used when the matrix is in implicit
     * building mode.
     *
     * A reference to entry (row, col) of the matrix is returned.
     * If entry (row, col) is accessed for the first time, it is created
     * on the fly.
     *
     * This method can only be used while building the matrix,
     * after compression operator[] gives a much better performance.
     */
    B& entry(size_type row, size_type col)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (build_mode!=implicit)
        DUNE_THROW(BCRSMatrixError,"requires implicit build mode");
      if (ready==built)
        DUNE_THROW(BCRSMatrixError,"matrix already built up, use operator[] for entry access now");
      if (ready==notAllocated)
        DUNE_THROW(BCRSMatrixError,"matrix size not set and no memory allocated yet");
      if (ready!=building)
        DUNE_THROW(InvalidStateException,"You may only use entry() during the 'building' stage");

      if (row >= n)
        DUNE_THROW(BCRSMatrixError,"row index exceeds matrix size");
      if (col >= m)
        DUNE_THROW(BCRSMatrixError,"column index exceeds matrix size");
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

        //do simultaneous operations on data array a
        std::ptrdiff_t offset = end - r[row].getindexptr();
        B* apos = r[row].getptr() + offset;

        //increase rowsize
        r[row].setsize(r[row].getsize()+1);

        //return reference to the newly created entry
        return *apos;
      }
    }

    //! Finishes the buildstage in implicit mode.
    /**
     * Performs compression of index and data arrays with linear
     * complexity in the number of nonzeroes.
     *
     * After calling this method, the matrix is in the built state
     * and no more entries can be added.
     *
     * \returns An object with some statistics about the compression for
     *          future optimization.
     */
    CompressionStatistics compress()
    {
      if (build_mode!=implicit)
        DUNE_THROW(BCRSMatrixError,"requires implicit build mode");
      if (ready==built)
        DUNE_THROW(BCRSMatrixError,"matrix already built up, no more need for compression");
      if (ready==notAllocated)
        DUNE_THROW(BCRSMatrixError,"matrix size not set and no memory allocated yet");
      if (ready!=building)
        DUNE_THROW(InvalidStateException,"You may only call compress() at the end of the 'building' stage");

      //calculate statistics
      CompressionStatistics stats;
      stats.overflow_total = overflow.size();
      stats.maximum = 0;

      //get insertion iterators pointing to one before start (for later use of ++it)
      size_type* jiit = j_.get();
      B* aiit = a;

      //get iterator to the smallest overflow element
      typename OverflowType::iterator oit = overflow.begin();

      //store a copy of index pointers on which to perform sorting
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
        r[i].setindexptr(jiit);
        r[i].setptr(aiit);

        for (it = perm.begin(); it != perm.end(); ++it)
        {
          //check whether there are elements in the overflow area which take precedence
          while ((oit!=overflow.end()) && (oit->first < std::make_pair(i,**it)))
          {
            //check whether there is enough memory to write to
            if (jiit > begin)
              DUNE_THROW(Dune::ImplicitModeCompressionBufferExhausted,
                         "Allocated memory for BCRSMatrix exhausted during compress()!"
                         "Please increase either the average number of entries per row or the compressionBufferSize value."
                         );
            //copy an element from the overflow area to the insertion position in a and j
            *jiit = oit->first.second;
            ++jiit;
            *aiit = oit->second;
            ++aiit;
            ++oit;
            r[i].setsize(r[i].getsize()+1);
          }

          //check whether there is enough memory to write to
          if (jiit > begin)
              DUNE_THROW(Dune::ImplicitModeCompressionBufferExhausted,
                         "Allocated memory for BCRSMatrix exhausted during compress()!"
                         "Please increase either the average number of entries per row or the compressionBufferSize value."
                         );

          //copy element from array
          *jiit = **it;
          ++jiit;
          B* apos = *it - j_.get() + a;
          *aiit = *apos;
          ++aiit;
        }

        //copy remaining elements from the overflow area
        while ((oit!=overflow.end()) && (oit->first.first == i))
        {
          //check whether there is enough memory to write to
          if (jiit > begin)
              DUNE_THROW(Dune::ImplicitModeCompressionBufferExhausted,
                         "Allocated memory for BCRSMatrix exhausted during compress()!"
                         "Please increase either the average number of entries per row or the compressionBufferSize value."
                         );

          //copy and element from the overflow area to the insertion position in a and j
          *jiit = oit->first.second;
          ++jiit;
          *aiit = oit->second;
          ++aiit;
          ++oit;
          r[i].setsize(r[i].getsize()+1);
        }

        // update maximum row size
        if (r[i].getsize()>stats.maximum)
          stats.maximum = r[i].getsize();
      }

      // overflow area may be cleared
      overflow.clear();

      //determine average number of entries and memory usage
      std::ptrdiff_t diff = (r[n-1].getindexptr() + r[n-1].getsize() - j_.get());
      nnz_ = diff;
      stats.avg = (double) (nnz_) / (double) n;
      stats.mem_ratio = (double) (nnz_) / (double) allocationSize_;

      //matrix is now built
      ready = built;

      return stats;
    }

    //===== vector space arithmetic

    //! vector space multiplication with scalar
    BCRSMatrix& operator*= (const field_type& k)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
#endif

      if (nnz_ > 0)
      {
        // process 1D array
        for (size_type i=0; i<nnz_; i++)
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
#ifdef DUNE_ISTL_WITH_CHECKING
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
#endif

      if (nnz_ > 0)
      {
        // process 1D array
        for (size_type i=0; i<nnz_; i++)
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
      if (ready != built || b.ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
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

    /*! \brief Subtract the entries of another matrix from this one.
     *
     * \param b The matrix to subtract from this one. Its sparsity pattern
     * has to be subset of the sparsity pattern of this matrix.
     */
    BCRSMatrix& operator-= (const BCRSMatrix& b)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (ready != built || b.ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
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
      if (ready != built || b.ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
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
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
      if (x.N()!=M()) DUNE_THROW(BCRSMatrixError,
                                 "Size mismatch: M: " << N() << "x" << M() << " x: " << x.N());
      if (y.N()!=N()) DUNE_THROW(BCRSMatrixError,
                                 "Size mismatch: M: " << N() << "x" << M() << " y: " << y.N());
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=this->begin(); i!=endi; ++i)
      {
        y[i.index()]=0;
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
        {
          auto&& xj = Impl::asVector(x[j.index()]);
          auto&& yi = Impl::asVector(y[i.index()]);
          Impl::asMatrix(*j).umv(xj, yi);
        }
      }
    }

    //! y += A x
    template<class X, class Y>
    void umv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
      if (x.N()!=M()) DUNE_THROW(BCRSMatrixError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(BCRSMatrixError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=this->begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
        {
          auto&& xj = Impl::asVector(x[j.index()]);
          auto&& yi = Impl::asVector(y[i.index()]);
          Impl::asMatrix(*j).umv(xj,yi);
        }
      }
    }

    //! y -= A x
    template<class X, class Y>
    void mmv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
      if (x.N()!=M()) DUNE_THROW(BCRSMatrixError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(BCRSMatrixError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=this->begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
        {
          auto&& xj = Impl::asVector(x[j.index()]);
          auto&& yi = Impl::asVector(y[i.index()]);
          Impl::asMatrix(*j).mmv(xj,yi);
        }
      }
    }

    //! y += alpha A x
    template<class X, class Y, class F>
    void usmv (F&& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
      if (x.N()!=M()) DUNE_THROW(BCRSMatrixError,"index out of range");
      if (y.N()!=N()) DUNE_THROW(BCRSMatrixError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=this->begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
        {
          auto&& xj = Impl::asVector(x[j.index()]);
          auto&& yi = Impl::asVector(y[i.index()]);
          Impl::asMatrix(*j).usmv(alpha,xj,yi);
        }
      }
    }

    //! y = A^T x
    template<class X, class Y>
    void mtv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
      if (x.N()!=N()) DUNE_THROW(BCRSMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(BCRSMatrixError,"index out of range");
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
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
      if (x.N()!=N()) DUNE_THROW(BCRSMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(BCRSMatrixError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=this->begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
        {
          auto&& xi = Impl::asVector(x[i.index()]);
          auto&& yj = Impl::asVector(y[j.index()]);
          Impl::asMatrix(*j).umtv(xi,yj);
        }
      }
    }

    //! y -= A^T x
    template<class X, class Y>
    void mmtv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (x.N()!=N()) DUNE_THROW(BCRSMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(BCRSMatrixError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=this->begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
        {
          auto&& xi = Impl::asVector(x[i.index()]);
          auto&& yj = Impl::asVector(y[j.index()]);
          Impl::asMatrix(*j).mmtv(xi,yj);
        }
      }
    }

    //! y += alpha A^T x
    template<class X, class Y>
    void usmtv (const field_type& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
      if (x.N()!=N()) DUNE_THROW(BCRSMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(BCRSMatrixError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=this->begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
        {
          auto&& xi = Impl::asVector(x[i.index()]);
          auto&& yj = Impl::asVector(y[j.index()]);
          Impl::asMatrix(*j).usmtv(alpha,xi,yj);
        }
      }
    }

    //! y += A^H x
    template<class X, class Y>
    void umhv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
      if (x.N()!=N()) DUNE_THROW(BCRSMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(BCRSMatrixError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=this->begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
        {
          auto&& xi = Impl::asVector(x[i.index()]);
          auto&& yj = Impl::asVector(y[j.index()]);
          Impl::asMatrix(*j).umhv(xi,yj);
        }
      }
    }

    //! y -= A^H x
    template<class X, class Y>
    void mmhv (const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
      if (x.N()!=N()) DUNE_THROW(BCRSMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(BCRSMatrixError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=this->begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
        {
          auto&& xi = Impl::asVector(x[i.index()]);
          auto&& yj = Impl::asVector(y[j.index()]);
          Impl::asMatrix(*j).mmhv(xi,yj);
        }
      }
    }

    //! y += alpha A^H x
    template<class X, class Y>
    void usmhv (const field_type& alpha, const X& x, Y& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
      if (x.N()!=N()) DUNE_THROW(BCRSMatrixError,"index out of range");
      if (y.N()!=M()) DUNE_THROW(BCRSMatrixError,"index out of range");
#endif
      ConstRowIterator endi=end();
      for (ConstRowIterator i=this->begin(); i!=endi; ++i)
      {
        ConstColIterator endj = (*i).end();
        for (ConstColIterator j=(*i).begin(); j!=endj; ++j)
        {
          auto&& xi = Impl::asVector(x[i.index()]);
          auto&& yj = Impl::asVector(y[j.index()]);
          Impl::asMatrix(*j).usmhv(alpha,xi,yj);
        }
      }
    }


    //===== norms

    //! square of frobenius norm, need for block recursion
    typename FieldTraits<field_type>::real_type frobenius_norm2 () const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");
#endif

      typename FieldTraits<field_type>::real_type sum=0;

      for (auto&& row : (*this))
        for (auto&& entry : row)
          sum += Impl::asMatrix(entry).frobenius_norm2();

      return sum;
    }

    //! frobenius norm: sqrt(sum over squared values of entries)
    typename FieldTraits<field_type>::real_type frobenius_norm () const
    {
      return sqrt(frobenius_norm2());
    }

    //! infinity norm (row sum norm, how to generalize for blocks?)
    template <typename ft = field_type,
              typename std::enable_if<!HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm() const {
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");

      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;

      real_type norm = 0;
      for (auto const &x : *this) {
        real_type sum = 0;
        for (auto const &y : x)
          sum += Impl::asMatrix(y).infinity_norm();
        norm = max(sum, norm);
      }
      return norm;
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    template <typename ft = field_type,
              typename std::enable_if<!HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm_real() const {
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");

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
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");

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
      if (ready != built)
        DUNE_THROW(BCRSMatrixError,"You can only call arithmetic operations on fully built BCRSMatrix instances");

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
      // in case of row-wise allocation
      if( nnz_ <= 0 )
        nnz_ = std::accumulate( begin(), end(), size_type( 0 ), [] ( size_type s, const row_type &row ) { return s+row.getsize(); } );
      return nnz_;
    }

    //! The current build stage of the matrix.
    BuildStage buildStage() const
    {
      return ready;
    }

    //! The currently selected build mode of the matrix.
    BuildMode buildMode() const
    {
      return build_mode;
    }

    //===== query

    //! return true if (i,j) is in pattern
    bool exists (size_type i, size_type j) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (i<0 || i>=n) DUNE_THROW(BCRSMatrixError,"row index out of range");
      if (j<0 || j>=m) DUNE_THROW(BCRSMatrixError,"column index out of range");
#endif
      return (r[i].size() && r[i].find(j) != r[i].end());
    }


  protected:
    // state information
    BuildMode build_mode;     // row wise or whole matrix
    BuildStage ready;               // indicate the stage the matrix building is in

    // The allocator used for memory management
    typename std::allocator_traits<A>::template rebind_alloc<B> allocator_;

    typename std::allocator_traits<A>::template rebind_alloc<row_type> rowAllocator_;

    typename std::allocator_traits<A>::template rebind_alloc<size_type> sizeAllocator_;

    // size of the matrix
    size_type n;       // number of rows
    size_type m;       // number of columns
    mutable size_type nnz_;     // number of nonzeroes contained in the matrix
    size_type allocationSize_; //allocated size of a and j arrays, except in implicit mode: nnz_==allocationsSize_
    // zero means that memory is allocated separately for each row.

    // the rows are dynamically allocated
    row_type* r;     // [n] the individual rows having pointers into a,j arrays

    // dynamically allocated memory
    B*   a;      // [allocationSize] non-zero entries of the matrix in row-wise ordering
    // If a single array of column indices is used, it can be shared
    // between different matrices with the same sparsity pattern
    std::shared_ptr<size_type> j_;  // [allocationSize] column indices of entries

    // additional data is needed in implicit buildmode
    size_type avg;
    double compressionBufferSize_;

    typedef std::map<std::pair<size_type,size_type>, B> OverflowType;
    OverflowType overflow;

    void setWindowPointers(ConstRowIterator row)
    {
      row_type current_row(a,j_.get(),0); // Pointers to current row data
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
          r[i].set(0,nullptr,nullptr);
        }
      }
    }

    //! Copy row sizes from iterator range starting at row and set column index pointers for all rows.
    /**
     * This method does not modify the data pointers, as those are set only
     * after building the pattern (to allow for a delayed allocation).
     */
    void setColumnPointers(ConstRowIterator row)
    {
      size_type* jptr = j_.get();
      for (size_type i=0; i<n; ++i, ++row) {
        // set row i
        size_type s = row->getsize();

        if (s>0) {
          // setup pointers and size
          r[i].setsize(s);
          r[i].setindexptr(jptr);
        } else{
          // empty row
          r[i].set(0,nullptr,nullptr);
        }

        // advance position in global array
        jptr += s;
      }
    }

    //! Set data pointers for all rows.
    /**
     * This method assumes that column pointers and row sizes have been correctly set up
     * by a prior call to setColumnPointers().
     */
    void setDataPointers()
    {
      B* aptr = a;
      for (size_type i=0; i<n; ++i) {
        // set row i
        if (r[i].getsize() > 0) {
          // setup pointers and size
          r[i].setptr(aptr);
        } else{
          // empty row
          r[i].set(0,nullptr,nullptr);
        }

        // advance position in global array
        aptr += r[i].getsize();
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

      if (notAllocated)
        return;

      if (allocationSize_>0)
      {
        // a,j_ have been allocated as one long vector
        j_.reset();
        if (a)
          {
            for(B *aiter=a+(allocationSize_-1), *aend=a-1; aiter!=aend; --aiter)
              std::allocator_traits<decltype(allocator_)>::destroy(allocator_, aiter);
            allocator_.deallocate(a,allocationSize_);
            a = nullptr;
          }
      }
      else if (r)
      {
        // check if memory for rows have been allocated individually
        for (size_type i=0; i<n; i++)
          if (r[i].getsize()>0)
          {
            for (B *col=r[i].getptr()+(r[i].getsize()-1),
                 *colend = r[i].getptr()-1; col!=colend; --col) {
              std::allocator_traits<decltype(allocator_)>::destroy(allocator_, col);
            }
            sizeAllocator_.deallocate(r[i].getindexptr(),1);
            allocator_.deallocate(r[i].getptr(),1);
            // clear out row data in case we don't want to deallocate the rows
            // otherwise we might run into a double free problem here later
            r[i].set(0,nullptr,nullptr);
          }
      }

      // deallocate the rows
      if (n>0 && deallocateRows && r) {
        for(row_type *riter=r+(n-1), *rend=r-1; riter!=rend; --riter)
          std::allocator_traits<decltype(rowAllocator_)>::destroy(rowAllocator_, riter);
        rowAllocator_.deallocate(r,n);
        r = nullptr;
      }

      // Mark matrix as not built at all.
      ready=notAllocated;

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
     * @param allocationSize The number of nonzero entries the matrix should hold (if omitted
     * defaults to 0).
     * @param allocateRow Whether we have to allocate the row pointers, too. (Defaults to
     * true)
     */
    void allocate(size_type rows, size_type columns, size_type allocationSize, bool allocateRows, bool allocate_data)
    {
      // Store size
      n = rows;
      m = columns;
      nnz_ = allocationSize;
      allocationSize_ = allocationSize;

      // allocate rows
      if(allocateRows) {
        if (n>0) {
          if (r)
            DUNE_THROW(InvalidStateException,"Rows have already been allocated, cannot allocate a second time");
          r = rowAllocator_.allocate(rows);
          // initialize row entries
          for(row_type* ri=r; ri!=r+rows; ++ri)
            std::allocator_traits<decltype(rowAllocator_)>::construct(rowAllocator_, ri, row_type());
        }else{
          r = 0;
        }
      }

      // allocate a and j_ array
      if (allocate_data)
        allocateData();
      // allocate column indices only if not yet present (enable sharing)
      if (allocationSize_>0) {
        // we copy allocator and size to the deleter since _j may outlive this class
        if (!j_.get())
          j_.reset(sizeAllocator_.allocate(allocationSize_),
            [alloc = sizeAllocator_, size = allocationSize_](auto ptr) mutable {
              alloc.deallocate(ptr, size);
            });
      }else{
        j_.reset();
      }

      // Mark the matrix as not built.
      ready = building;
    }

    void allocateData()
    {
      if (a)
        DUNE_THROW(InvalidStateException,"Cannot allocate data array (already allocated)");
      if (allocationSize_>0) {
        a = allocator_.allocate(allocationSize_);
        // use placement new to call constructor that allocates
        // additional memory.
        new (a) B[allocationSize_];
      } else {
        a = nullptr;
      }
    }

    /** @brief organizes allocation implicit mode
     * calculates correct array size to be allocated and sets the
     * the window pointers to their correct positions for insertion.
     * internally uses allocate() for the real allocation.
     */
    void implicit_allocate(size_type _n, size_type _m)
    {
      if (build_mode != implicit)
        DUNE_THROW(InvalidStateException,"implicit_allocate() may only be called in implicit build mode");
      if (ready != notAllocated)
        DUNE_THROW(InvalidStateException,"memory has already been allocated");

      // check to make sure the user has actually set the parameters
      if (compressionBufferSize_ < 0)
        DUNE_THROW(InvalidStateException,"You have to set the implicit build mode parameters before starting to build the matrix");
      //calculate size of overflow region, add buffer for row sort!
      size_type osize = (size_type) (_n*avg)*compressionBufferSize_ + 4*avg;
      allocationSize_ = _n*avg + osize;

      allocate(_n, _m, allocationSize_,true,true);

      //set row pointers correctly
      size_type* jptr = j_.get() + osize;
      B* aptr = a + osize;
      for (size_type i=0; i<n; i++)
      {
        r[i].set(0,aptr,jptr);
        jptr = jptr + avg;
        aptr = aptr + avg;
      }

      ready = building;
    }
  };


  template<class B, class A>
  struct FieldTraits< BCRSMatrix<B, A> >
  {
    using field_type = typename BCRSMatrix<B, A>::field_type;
    using real_type = typename FieldTraits<field_type>::real_type;
  };

  /** @} end documentation */

} // end namespace

#endif
