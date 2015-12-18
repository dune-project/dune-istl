// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_PRECONDITIONERS_HH
#define DUNE_ISTL_PRECONDITIONERS_HH

#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <string>

#include <dune/common/unused.hh>

#include "preconditioner.hh"
#include "solver.hh"
#include "solvercategory.hh"
#include "istlexception.hh"
#include "matrixutils.hh"
#include "gsetc.hh"
#include "ilu.hh"


namespace Dune {
  /** @defgroup ISTL_Prec Preconditioners
   * @ingroup ISTL_Solvers
   *
   * All of our \ref ISTL_Solvers "Krylow solvers" are preconditioned versions.
   * There are sequential preconditioners (e,g. SeqJacobi, SeqSOR, SeqSSOR) as well as parallel preconditioners
   * (e.g. AMG, BlockPreconditioner) available for plugging them into the solvers
   * together with matching ScalarProducts.
   *
   * Some of the available perconditioners (e.g. SeqJacobi, SeqSOR, SeqSSOR))
   * may be given an aditional int as a template parameter, the block recursion level.
   * These preconditioners
   * can be used on blockrecursive matrices with an arbitrary hierarchy depths
   * (eg. BCRSMatrix<BCRSMatrix<FieldMatrix,n,m> > >. Given a block recursion level
   * \f$k\f$ those preconditioners work as
   * normal on the offdiagonal blocks, treating them as traditional matrix
   * entries. For the diagonal values a special procedure applies:  If
   * \f$k>1\f$ the diagonal is treated as a matrix itself and the preconditioner
   * is applied recursively on the matrix representing the diagonal value
   * \f$D=A_{ii}\f$ with block level \f$k-1\f$. For the case that \f$k=1\f$ the diagonal
   * is treated as a
   * matrix entry resulting in a linear solve or an identity operation
   * depending on the algorithm.
   */

  /** @addtogroup ISTL_Prec
          @{
   */
  /** \file

     \brief    Define general preconditioner interface

     Wrap the methods implemented by ISTL in this interface.
     However, the interface is extensible such that new preconditioners
     can be implemented and used with the solvers.
   */



  /**
   * @brief Turns an InverseOperator into a Preconditioner.
   * @tparam O The type of the inverse operator to wrap.
   */
  template<class O, int c>
  class InverseOperator2Preconditioner :
    public Preconditioner<typename O::domain_type, typename O::range_type>
  {
  public:
    //! \brief The domain type of the preconditioner.
    typedef typename O::domain_type domain_type;
    //! \brief The range type of the preconditioner.
    typedef typename O::range_type range_type;
    //! \brief The field type of the preconditioner.
    typedef typename range_type::field_type field_type;
    typedef O InverseOperator;

    // define the category
    enum {
      //! \brief The category the preconditioner is part of.
      category=c
    };

    /**
     * @brief Construct the preconditioner from the solver
     * @param inverse_operator The inverse operator to wrap.
     */
    InverseOperator2Preconditioner(InverseOperator& inverse_operator)
    : inverse_operator_(inverse_operator)
    {}

    void pre(domain_type&,range_type&)
    {}

    void apply(domain_type& v, const range_type& d)
    {
      InverseOperatorResult res;
      range_type copy(d);
      inverse_operator_.apply(v, copy, res);
    }

    void post(domain_type&)
    {}

  private:
    InverseOperator& inverse_operator_;
  };

  //=====================================================================
  // Implementation of this interface for sequential ISTL-preconditioners
  //=====================================================================


  /*!
     \brief Sequential SSOR preconditioner.

     Wraps the naked ISTL generic SSOR preconditioner into the
      solver framework.

     \tparam M The matrix type to operate on
     \tparam X Type of the update
     \tparam Y Type of the defect
     \tparam l The block level to invert. Default is 1
   */
  template<class M, class X, class Y, int l=1>
  class SeqSSOR : public Preconditioner<X,Y> {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef M matrix_type;
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

       constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
       \param n The number of iterations to perform.
       \param w The relaxation factor.
     */
    SeqSSOR (const M& A, int n, field_type w)
      : _A_(A), _n(n), _w(w)
    {
      CheckIfDiagonalPresent<M,l>::check(_A_);
    }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b)
    {
      DUNE_UNUSED_PARAMETER(x);
      DUNE_UNUSED_PARAMETER(b);

    }

    /*!
       \brief Apply the preconditioner

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      for (int i=0; i<_n; i++) {
        bsorf(_A_,v,d,_w,BL<l>());
        bsorb(_A_,v,d,_w,BL<l>());
      }
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x)
    {
      DUNE_UNUSED_PARAMETER(x);
    }

  private:
    //! \brief The matrix we operate on.
    const M& _A_;
    //! \brief The number of steps to do in apply
    int _n;
    //! \brief The relaxation factor to use
    field_type _w;
  };



  /*!
     \brief Sequential SOR preconditioner.

     Wraps the naked ISTL generic SOR preconditioner into the
     solver framework.

     \tparam M The matrix type to operate on
     \tparam X Type of the update
     \tparam Y Type of the defect
     \tparam l The block level to invert. Default is 1
   */
  template<class M, class X, class Y, int l=1>
  class SeqSOR : public Preconditioner<X,Y> {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef M matrix_type;
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

       constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
       \param n The number of iterations to perform.
       \param w The relaxation factor.
     */
    SeqSOR (const M& A, int n, field_type w)
      : _A_(A), _n(n), _w(w)
    {
      CheckIfDiagonalPresent<M,l>::check(_A_);
    }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b)
    {
      DUNE_UNUSED_PARAMETER(x);
      DUNE_UNUSED_PARAMETER(b);
    }

    /*!
       \brief Apply the preconditioner.

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      this->template apply<true>(v,d);
    }

    /*!
       \brief Apply the preconditioner in a special direction.

       The template parameter forward indications the direction
       the smoother is applied. If true The application is
       started at the lowest index in the vector v, if false at
       the highest index of vector v.
     */
    template<bool forward>
    void apply(X& v, const Y& d)
    {
      if(forward)
        for (int i=0; i<_n; i++) {
          bsorf(_A_,v,d,_w,BL<l>());
        }
      else
        for (int i=0; i<_n; i++) {
          bsorb(_A_,v,d,_w,BL<l>());
        }
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x)
    {
      DUNE_UNUSED_PARAMETER(x);
    }

  private:
    //! \brief the matrix we operate on.
    const M& _A_;
    //! \brief The number of steps to perform in apply.
    int _n;
    //! \brief The relaxation factor to use.
    field_type _w;
  };


  /*! \brief Sequential Gauss Seidel preconditioner

     Wraps the naked ISTL generic block Gauss-Seidel preconditioner into the
      solver framework.

     \tparam M The matrix type to operate on
     \tparam X Type of the update
     \tparam Y Type of the defect
     \tparam l The block level to invert. Default is 1
   */
  template<class M, class X, class Y, int l=1>
  class SeqGS : public Preconditioner<X,Y> {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef M matrix_type;
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
       \param A The matrix to operate on.
       \param n The number of iterations to perform.
       \param w The relaxation factor.
     */
    SeqGS (const M& A, int n, field_type w)
      : _A_(A), _n(n), _w(w)
    {
      CheckIfDiagonalPresent<M,l>::check(_A_);
    }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b)
    {
      DUNE_UNUSED_PARAMETER(x);
      DUNE_UNUSED_PARAMETER(b);
    }

    /*!
       \brief Apply the preconditioner.

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      for (int i=0; i<_n; i++) {
        dbgs(_A_,v,d,_w,BL<l>());
      }
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x)
    {
      DUNE_UNUSED_PARAMETER(x);
    }

  private:
    //! \brief The matrix we operate on.
    const M& _A_;
    //! \brief The number of iterations to perform in apply.
    int _n;
    //! \brief The relaxation factor to use.
    field_type _w;
  };


  /*! \brief The sequential jacobian preconditioner.

     Wraps the naked ISTL generic block Jacobi preconditioner into the
      solver framework.

     \tparam M The matrix type to operate on
     \tparam X Type of the update
     \tparam Y Type of the defect
     \tparam l The block level to invert. Default is 1
   */
  template<class M, class X, class Y, int l=1>
  class SeqJac : public Preconditioner<X,Y> {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef M matrix_type;
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;
    //! \brief The field type of the preconditioner.
    typedef typename X::field_type field_type;

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
    SeqJac (const M& A, int n, field_type w)
      : _A_(A), _n(n), _w(w)
    {
      CheckIfDiagonalPresent<M,l>::check(_A_);
    }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b)
    {
      DUNE_UNUSED_PARAMETER(x);
      DUNE_UNUSED_PARAMETER(b);
    }

    /*!
       \brief Apply the preconditioner.

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      for (int i=0; i<_n; i++) {
        dbjac(_A_,v,d,_w,BL<l>());
      }
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x)
    {
      DUNE_UNUSED_PARAMETER(x);
    }

  private:
    //! \brief The matrix we operate on.
    const M& _A_;
    //! \brief The number of steps to perform during apply.
    int _n;
    //! \brief The relaxation parameter to use.
    field_type _w;
  };



  /*!
     \brief Sequential ILU0 preconditioner.

     Wraps the naked ISTL generic ILU0 preconditioner into the solver framework.

     \tparam M The matrix type to operate on
     \tparam X Type of the update
     \tparam Y Type of the defect
     \tparam l Ignored. Just there to have the same number of template arguments
     as other preconditioners.
   */
  template<class M, class X, class Y, int l=1>
  class SeqILU0 : public Preconditioner<X,Y> {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef typename std::remove_const<M>::type matrix_type;
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
       \param A The matrix to operate on.
       \param w The relaxation factor.
     */
    SeqILU0 (const M& A, field_type w)
      : ILU(A) // copy A
    {
      _w =w;
      bilu0_decomposition(ILU);
    }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b)
    {
      DUNE_UNUSED_PARAMETER(x);
      DUNE_UNUSED_PARAMETER(b);
    }

    /*!
       \brief Apply the preconditoner.

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      bilu_backsolve(ILU,v,d);
      v *= _w;
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x)
    {
      DUNE_UNUSED_PARAMETER(x);
    }

  private:
    //! \brief The relaxation factor to use.
    field_type _w;
    //! \brief The ILU0 decomposition of the matrix.
    matrix_type ILU;
  };


  /*!
     \brief Sequential ILU(n) preconditioner.

     Wraps the naked ISTL generic ILU(n) preconditioner into the
     solver framework.


     \tparam M The matrix type to operate on
     \tparam X Type of the update
     \tparam Y Type of the defect
     \tparam l Ignored. Just there to have the same number of template arguments
     as other preconditioners.
   */
  template<class M, class X, class Y, int l=1>
  class SeqILUn : public Preconditioner<X,Y> {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef typename std::remove_const<M>::type matrix_type;
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
       \param A The matrix to operate on.
       \param n The number of iterations to perform.
       \param w The relaxation factor.
     */
    SeqILUn (const M& A, int n, field_type w)
      : ILU(A.N(),A.M(),M::row_wise)
    {
      _n = n;
      _w = w;
      bilu_decomposition(A,n,ILU);
    }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b)
    {
      DUNE_UNUSED_PARAMETER(x);
      DUNE_UNUSED_PARAMETER(b);
    }

    /*!
       \brief Apply the precondioner.

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      bilu_backsolve(ILU,v,d);
      v *= _w;
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x)
    {
      DUNE_UNUSED_PARAMETER(x);
    }

  private:
    //! \brief ILU(n) decomposition of the matrix we operate on.
    matrix_type ILU;
    //! \brief The number of steps to perform in apply.
    int _n;
    //! \brief The relaxation factor to use.
    field_type _w;
  };



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
    virtual void pre (X& x, Y& b)
    {
      DUNE_UNUSED_PARAMETER(x);
      DUNE_UNUSED_PARAMETER(b);
    }

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
    virtual void post (X& x)
    {
      DUNE_UNUSED_PARAMETER(x);
    }

  private:
    //! \brief The relaxation factor to use.
    field_type _w;
  };



  /** @} end documentation */

} // end namespace

#endif
