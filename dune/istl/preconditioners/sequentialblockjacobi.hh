// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_PRECONDITIONERS_SEQUENTIALBLOCKJACOBI_HH
#define DUNE_ISTL_PRECONDITIONERS_SEQUENTIALBLOCKJACOBI_HH

#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <string>

#include <dune/common/kernel/bell.hh>
#include <dune/common/kernel/blockdiagonal.hh>

#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solvercategory.hh>
#include <dune/istl/istlexception.hh>
#include <dune/istl/bellmatrix/host.hh>
#include <dune/istl/blockvector/host.hh>

namespace Dune {
  namespace ISTL {

  /** @addtogroup ISTL_Prec
          @{
   */
  /** \file

     \brief    Define general preconditioner interface

     Wrap the methods implemented by ISTL in this interface.
     However, the interface is extensible such that new preconditioners
     can be implemented and used with the solvers.
   */




  /*! \brief The sequential jacobian preconditioner.

     Wraps the naked ISTL generic block Jacobi preconditioner into the
      solver framework.

     \tparam M The matrix type to operate on
     \tparam X Type of the update
     \tparam Y Type of the defect
   */
  template<
    typename M,
    typename X,
    typename Y,
    typename D_ = typename Memory::allocator_domain<typename M::Allocator>::type // memory domain
    >
  class SequentialBlockJacobi : public Preconditioner<X,Y> {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef M Matrix;
    //! \brief The domain type of the preconditioner.
    typedef X Domain;
    //! \brief The range type of the preconditioner.
    typedef Y Range;
    //! \brief The field type of the preconditioner.
    typedef typename X::value_type value_type;

    // TODO: Make sure the allocators are compatible

    typedef typename X::value_type domain_value_type;
    typedef typename Y::value_type range_value_type;
    typedef typename Matrix::value_type matrix_value_type;
    typedef typename Matrix::Allocator::size_type size_type;

    static const size_type kernel_block_size = Matrix::Allocator::block_size;
    static const size_type alignment = Matrix::Allocator::alignment;

    typedef typename Matrix::range_type range_type;

    // define the category
    enum {
      //! \brief The category the preconditioner is part of
      category=SolverCategory::sequential
    };

    /*! \brief Constructor.

       Constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
       \param n The number of iterations to perform.
       \param w The relaxation factor.
     */
    SequentialBlockJacobi(const M& A, value_type w, bool pivot, size_type iterations = 3)
      : _A(A)
      , _v_new(A.rows(),A.blockRows())
      , _mat(nullptr)
      , _permutation(nullptr)
      , _w(w)
      , _iterations(iterations)
      , _pivot(pivot)
      , _factorized(false)
      , _mat_allocation_size(0)
      , _permutation_allocation_size(0)
    {
      computeLUFactorization();
    }

    virtual ~SequentialBlockJacobi()
    {
      deallocate();
    }

    void allocate()
    {
      if (_mat)
        DUNE_THROW(Exception,"not supported");
      _mat_allocation_size = _A.layout().allocatedRows()*_A.blockRows()*_A.blockCols();
      _mat = _A.allocator().allocate(_mat_allocation_size);
      if (_pivot)
        {
          _permutation_allocation_size = _A.layout().allocatedRows()*_A.blockRows();
          _permutation = _A.layout().allocator().allocate(_permutation_allocation_size);
        }
    }

    void deallocate()
    {
      if (_mat)
        {
          _A.allocator().deallocate(_mat,_mat_allocation_size);
          _mat = nullptr;
          _mat_allocation_size = 0;
        }
      if (_permutation)
        {
          _A.layout().allocator().deallocate(_permutation,_permutation_allocation_size);
          _permutation = nullptr;
          _permutation_allocation_size = 0;
        }
    }

    void computeLUFactorization()
    {
      // currently, this only works for square blocks
      const size_type block_rows = _A.blockRows();
      const size_type block_cols = _A.blockCols();
      assert(block_rows == block_cols);
      const size_type block_size = block_rows * block_cols;

      if (!_mat)
        allocate();

      // extract diagonal blocks
      tbb::parallel_for(
        _A.iteration_range(),
        [&](const range_type& r)
        {
          Dune::Kernel::block_diagonal::blocked::extract_bell_diagonal<
            matrix_value_type,
            matrix_value_type,
            size_type,
            alignment,
            kernel_block_size>(
              _mat + r.begin() * block_size,
              _A.data() + _A.layout().blockOffset(r.begin_block()) * block_size,
              _A.layout().colIndex() + _A.layout().blockOffset(r.begin_block()),
              _A.layout().blockOffset() + r.begin_block(),
              _A.layout().rowLength() + r.begin(),
              r.block_count(),
              _A.blockRows(),
              _A.blockCols(),
              r.begin());
        });

      // calculate LU decomposition
      tbb::parallel_for(
        _A.iteration_range(),
        [&](const range_type& r)
        {
          if (_pivot)
            Dune::Kernel::block_diagonal::blocked::lu_decomposition_partial_pivot<
              matrix_value_type,
              size_type,
              alignment,
              kernel_block_size>(
                _mat + r.begin() * block_size,
                _mat + r.begin() * block_size,
                _permutation + r.begin() * _A.blockRows(),
                r.block_count(),
                _A.blockRows());
          else
            Dune::Kernel::block_diagonal::blocked::lu_decomposition_no_pivot<
              matrix_value_type,
              size_type,
              alignment,
              kernel_block_size>(
                _mat + r.begin() * block_size,
                _mat + r.begin() * block_size,
                r.block_count(),
                _A.blockRows());
        });

      // reset padding rows to unit matrices to avoid division by zero in apply()
      size_type rows = _A.layout().rows();
      if (rows % kernel_block_size != 0)
        {
          size_type block = _A.layout().blocks() - 1;
          for (size_type row = _A.layout().rows() % kernel_block_size; row < kernel_block_size; ++row)
            for (size_type ii = 0; ii < block_rows; ++ii)
              for (size_type jj = 0; jj < block_cols; ++jj)
                _mat[block * kernel_block_size * block_size + ii * block_cols * kernel_block_size + jj * kernel_block_size + row] = ii == jj ? value_type(1) : value_type(0);
        }

      _factorized = true;
    }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b) {}


#if 0
    // old version of apply with full rhs update (including diagonal block)

    /*!
       \brief Apply the preconditioner.

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      assert(_factorized);

      size_type block_rows = _A.blockRows();
      size_type block_cols = _A.blockCols();
      size_type block_size = block_rows * block_cols;

      //int iterations = _iterations > 10 ? _iterations : 1;

      for (int i = 0; i < _iterations; ++i)
        {

          Y d_(d);
          _A.mmv(v,d_);

          tbb::parallel_for(
            _A.iteration_range(),
            [&](const range_type& r)
            {
              // allocate temporary variable for use inside kernel
              // We do this inside the lambda to get separate vectors for each thread
              domain_value_type* y = v.allocator().allocate(block_rows * kernel_block_size);

              if (_pivot)
                Dune::Kernel::block_diagonal::blocked::lu_solve_partial_pivot<
                  domain_value_type,
                  range_value_type,
                  matrix_value_type,
                  size_type,
                  alignment,
                  kernel_block_size>(
                    v.data() + r.begin() * block_rows,
                    d_.data() + r.begin() * block_rows,
                    _mat + r.begin() * block_size,
                    y,
                    _permutation + r.begin() * block_rows,
                    r.block_count(),
                    block_rows,
                    _w);
              else
                Dune::Kernel::block_diagonal::blocked::lu_solve_no_pivot<
                  domain_value_type,
                  range_value_type,
                  matrix_value_type,
                  size_type,
                  alignment,
                  kernel_block_size>(
                    v.data() + r.begin() * block_rows,
                    d_.data() + r.begin() * block_rows,
                    _mat + r.begin() * block_size,
                    y,
                    r.block_count(),
                    block_rows,
                    _w);


              // free temporary vector
              v.allocator().deallocate(y,block_rows * kernel_block_size);

            });
        }
    }

#else

    // new version of apply (skips diagonal block, better memory handling)

    /*!
       \brief Apply the preconditioner.

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      assert(_factorized);

      size_type block_rows = _A.blockRows();
      size_type block_cols = _A.blockCols();
      size_type block_size = block_rows * block_cols;

      //int iterations = _iterations > 10 ? _iterations : 1;

      for (size_type i = 0; i < _iterations; ++i)
        {
          tbb::parallel_for(
            _A.iteration_range(),
            [&](const range_type& r)
            {
              // allocate temporary variables for use inside kernel
              // We do this inside the lambda to get separate vectors for each thread
              domain_value_type* y = v.allocator().allocate(block_rows * kernel_block_size);
              domain_value_type* rhs = v.allocator().allocate(block_rows * kernel_block_size);

              if (_pivot)
                DUNE_THROW(NotImplemented,"not implemented yet");
              else
                Dune::Kernel::bell::preconditioners::blocked::jacobi_no_lu_pivot<
                  domain_value_type,
                  range_value_type,
                  matrix_value_type,
                  size_type,
                  alignment,
                  kernel_block_size>(
                    _v_new.data() + r.begin() * block_rows,
                    v.data(), // don't offset into the old data, we get absolute column indices out of the matrix
                    d.data() + r.begin() * block_rows,
                    _mat + r.begin() * block_size,
                    _A.data() + _A.layout().blockOffset(r.begin_block()) * block_size,
                    _A.layout().colIndex()+_A.layout().blockOffset(r.begin_block()),
                    _A.layout().blockOffset()+r.begin_block(),
                    y,
                    rhs,
                    r.block_count(),
                    block_rows,
                    r.begin(),
                    _w);


              // free temporary vectors
              v.allocator().deallocate(y,block_rows * kernel_block_size);
              v.allocator().deallocate(rhs,block_rows * kernel_block_size);

            });
          v.axpy(_w,_v_new);
        }
    }

#endif // 0

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x) {}

  private:

    //! \brief The matrix we operate on.
    const M& _A;

    //! Vector for temporary output storage
    Y _v_new;

    // Data arrays
    matrix_value_type* _mat;
    size_type* _permutation;

    //! \brief The relaxation parameter to use.
    value_type _w;
    //! The number of iterations per call to apply()
    size_type _iterations;

    bool _pivot;
    bool _factorized;

    size_type _mat_allocation_size;
    size_type _permutation_allocation_size;

  };




#if 0

  /*!
     \brief Richardson preconditioner.

        Multiply simply by a constant.

     \tparam X Type of the update
     \tparam Y Type of the defect
   */
  template<class X, class Y>
  class Richardson : public Preconditioner<X,Y> {
  public:
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;
    //! \brief The field type of the preconditioner.
    typedef typename X::field_type field_type;

    // define the category
    enum {
      //! \brief The category the preconditioner is part of.
      category=SolverCategory::sequential
    };

    /*! \brief Constructor.

       Constructor gets all parameters to operate the prec.
       \param w The relaxation factor.
     */
    Richardson (field_type w=1.0)
    {
      _w = w;
    }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b) {}

    /*!
       \brief Apply the precondioner.

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      v = d;
      v *= _w;
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x) {}

  private:
    //! \brief The relaxation factor to use.
    field_type _w;
  };

#endif

  /** @} end documentation */

  } // namespace ISTL
} // namespace Dune

#endif // DUNE_ISTL_PRECONDITIONERS_SEQUENTIALBLOCKJACOBI_HH
