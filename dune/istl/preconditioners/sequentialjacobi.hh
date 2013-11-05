// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_PRECONDITIONERS_SEQUENTIALJACOBI_HH
#define DUNE_ISTL_PRECONDITIONERS_SEQUENTIALJACOBI_HH

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



  // forward declaration that takes care of automatically extracting the memory domain
  template<
    typename M, // matrix
    typename X, // domain
    typename Y, // range
    typename D_ = typename Memory::allocator_domain<typename M::Allocator>::type // memory domain
    >
  class SequentialJacobi;


  template<typename M, typename X, typename Y>
  class SequentialJacobi<M,X,Y,Memory::Domain::Host>
    : public Preconditioner<X,Y> {
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
    SequentialJacobi(const M& A, value_type w, size_type iterations = 3)
      : _A(A)
      , _v_new(A.rows())
      , _w(w)
      , _iterations(iterations)
    {}

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b) {}



    // new version of apply (skips diagonal block, better memory handling)

    /*!
       \brief Apply the preconditioner.

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      //int iterations = _iterations > 10 ? _iterations : 1;

      for (int i = 0; i < _iterations; ++i)
        {
          tbb::parallel_for(
            _A.iteration_range(),
            [&](const range_type& r)
            {
              // allocate temporary variables for use inside kernel
              // We do this inside the lambda to get separate vectors for each thread
              domain_value_type* diag = v.allocator().allocate(kernel_block_size);
              domain_value_type* rhs = v.allocator().allocate(kernel_block_size);

              Dune::Kernel::ell::preconditioners::blocked::jacobi<
                domain_value_type,
                range_value_type,
                matrix_value_type,
                size_type,
                alignment,
                kernel_block_size>(
                  _v_new.data() + r.begin(),
                  v.data(), // don't offset into the old data, we get absolute column indices out of the matrix
                  d.data() + r.begin(),
                  _A.data() + _A.layout().blockOffset(r.begin_block()),
                  _A.layout().colIndex()+_A.layout().blockOffset(r.begin_block()),
                  _A.layout().blockOffset()+r.begin_block(),
                  diag,
                  rhs,
                  r.block_count(),
                  r.begin(),
                  v.size(),
                  _w);


              // free temporary vectors
              v.allocator().deallocate(diag,kernel_block_size);
              v.allocator().deallocate(rhs,kernel_block_size);

            });
          v.axpy(_w,_v_new);
        }
    }

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

    //! \brief The relaxation parameter to use.
    value_type _w;
    //! The number of iterations per call to apply()
    size_type _iterations;

  };

  template<typename M, typename X, typename Y>
  class SequentialJacobi<M,X,Y,Memory::Domain::CUDA>
    : public Preconditioner<X,Y> {
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
    SequentialJacobi(const M& A, value_type w, size_type iterations = 3)
      : _A(A)
      , _v_new(A.rows())
      , _w(w)
      , _iterations(iterations)
    {}

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b) {}



    /*!
       \brief Apply the preconditioner.

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      //int iterations = _iterations > 10 ? _iterations : 1;

      for (size_type i (0) ; i < _iterations; ++i)
        {
          for (size_type j(0) ; j < d.size () ; ++j)
            _v_new(i, value_type(1) / _A(j, j));
          v.axpy(_w,_v_new);
        }
    }

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

    //! \brief The relaxation parameter to use.
    value_type _w;
    //! The number of iterations per call to apply()
    size_type _iterations;

  };

  /** @} end documentation */

  } // namespace ISTL
} // namespace Dune

#endif // DUNE_ISTL_PRECONDITIONERS_SEQUENTIALJACOBI_HH
