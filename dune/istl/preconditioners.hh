// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_PRECONDITIONERS_HH
#define DUNE_ISTL_PRECONDITIONERS_HH

#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>

#include <dune/common/simd/simd.hh>
#include <dune/common/parametertree.hh>

#include <dune/istl/solverregistry.hh>
#include "preconditioner.hh"
#include "solver.hh"
#include "solvercategory.hh"
#include "istlexception.hh"
#include "matrixutils.hh"
#include "gsetc.hh"
#include "ildl.hh"
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
   * Some of the available preconditioners (e.g. SeqJacobi, SeqSOR, SeqSSOR))
   * may be given an aditional int as a template parameter, the block recursion level.
   * These preconditioners
   * can be used on block-recursive matrices with an arbitrary hierarchy depth
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
  template<class O, int c = -1>
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
    //! \brief scalar type underlying the field_type
    typedef Simd::Scalar<field_type> scalar_field_type;
    //! \brief real scalar type underlying the field_type
    typedef typename FieldTraits<scalar_field_type>::real_type real_field_type;
    //! \brief type of the wrapped inverse operator
    typedef O InverseOperator;

    /**
     * @brief Construct the preconditioner from the solver
     * @param inverse_operator The inverse operator to wrap.
     */
    InverseOperator2Preconditioner(InverseOperator& inverse_operator)
    : inverse_operator_(inverse_operator)
    {
      if(c != -1 && SolverCategory::category(inverse_operator_) != c)
        DUNE_THROW(InvalidStateException, "User-supplied solver category does not match that of the given inverse operator");
    }

    virtual void pre(domain_type&,range_type&)
    {}

    virtual void apply(domain_type& v, const range_type& d)
    {
      InverseOperatorResult res;
      range_type copy(d);
      inverse_operator_.apply(v, copy, res);
    }

    virtual void post(domain_type&)
    {}

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::category(inverse_operator_);
    }

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
    //! \brief scalar type underlying the field_type
    typedef Simd::Scalar<field_type> scalar_field_type;
    //! \brief real scalar type underlying the field_type
    typedef typename FieldTraits<scalar_field_type>::real_type real_field_type;

    /*! \brief Constructor.

       constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
       \param n The number of iterations to perform.
       \param w The relaxation factor.
     */
    SeqSSOR (const M& A, int n, real_field_type w)
      : _A_(A), _n(n), _w(w)
    {
      CheckIfDiagonalPresent<M,l>::check(_A_);
    }

    /*!
       \brief Constructor.

       \param A The assembled linear operator to use.
       \param configuration ParameterTree containing preconditioner parameters.

       ParameterTree Key | Meaning
       ------------------|------------
       iterations        | The number of iterations to perform. default=1
       relaxation        | The relaxation factor. default=1.0

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    SeqSSOR (const std::shared_ptr<const AssembledLinearOperator<M,X,Y>>& A, const ParameterTree& configuration)
      : SeqSSOR(A->getmat(), configuration)
    {}

    /*!
       \brief Constructor.

       \param A The matrix to operate on.
       \param configuration ParameterTree containing preconditioner parameters.

       ParameterTree Key | Meaning
       ------------------|------------
       iterations        | The number of iterations to perform. default=1
       relaxation        | The relaxation factor. default=1.0

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    SeqSSOR (const M& A, const ParameterTree& configuration)
      : SeqSSOR(A, configuration.get<int>("iterations",1), configuration.get<real_field_type>("relaxation",1.0))
    {}

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre ([[maybe_unused]] X& x, [[maybe_unused]] Y& b)
    {}

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
    virtual void post ([[maybe_unused]] X& x)
    {}

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::sequential;
    }

  private:
    //! \brief The matrix we operate on.
    const M& _A_;
    //! \brief The number of steps to do in apply
    int _n;
    //! \brief The relaxation factor to use
    real_field_type _w;
  };
  DUNE_REGISTER_PRECONDITIONER("ssor", defaultPreconditionerBlockLevelCreator<Dune::SeqSSOR>());


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
    //! \brief scalar type underlying the field_type
    typedef Simd::Scalar<field_type> scalar_field_type;
    //! \brief real scalar type underlying the field_type
    typedef typename FieldTraits<scalar_field_type>::real_type real_field_type;

    /*! \brief Constructor.

       constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
       \param n The number of iterations to perform.
       \param w The relaxation factor.
     */
    SeqSOR (const M& A, int n, real_field_type w)
      : _A_(A), _n(n), _w(w)
    {
      CheckIfDiagonalPresent<M,l>::check(_A_);
    }

    /*!
       \brief Constructor.

       \param A The assembled linear operator to use.
       \param configuration ParameterTree containing preconditioner parameters.

       ParameterTree Key | Meaning
       ------------------|------------
       iterations        | The number of iterations to perform. default=1
       relaxation        | The relaxation factor. default=1.0

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    SeqSOR (const std::shared_ptr<const AssembledLinearOperator<M,X,Y>>& A, const ParameterTree& configuration)
      : SeqSOR(A->getmat(), configuration)
    {}

    /*!
       \brief Constructor.

       \param A The matrix to operate on.
       \param configuration ParameterTree containing preconditioner parameters.

       ParameterTree Key | Meaning
       ------------------|------------
       iterations        | The number of iterations to perform. default=1
       relaxation        | The relaxation factor. default=1.0

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    SeqSOR (const M& A, const ParameterTree& configuration)
      : SeqSOR(A, configuration.get<int>("iterations",1), configuration.get<real_field_type>("relaxation",1.0))
    {}

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre ([[maybe_unused]] X& x, [[maybe_unused]] Y& b)
    {}

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
    virtual void post ([[maybe_unused]] X& x)
    {}

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::sequential;
    }

  private:
    //! \brief the matrix we operate on.
    const M& _A_;
    //! \brief The number of steps to perform in apply.
    int _n;
    //! \brief The relaxation factor to use.
    real_field_type _w;
  };
  DUNE_REGISTER_PRECONDITIONER("sor", defaultPreconditionerBlockLevelCreator<Dune::SeqSOR>());


  /*! \brief Sequential Gauss Seidel preconditioner

     Wraps the naked ISTL generic block Gauss-Seidel preconditioner into the
      solver framework.

     \tparam M The matrix type to operate on
     \tparam X Type of the update
     \tparam Y Type of the defect
     \tparam l The block level to invert. Default is 1
   */
  template<class M, class X, class Y, int l=1>
  using SeqGS = SeqSOR<M,X,Y,l>;
  DUNE_REGISTER_PRECONDITIONER("gs", defaultPreconditionerBlockLevelCreator<Dune::SeqGS>());

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
    //! \brief scalar type underlying the field_type
    typedef Simd::Scalar<field_type> scalar_field_type;
    //! \brief real scalar type underlying the field_type
    typedef typename FieldTraits<scalar_field_type>::real_type real_field_type;

    /*! \brief Constructor.

       Constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
       \param n The number of iterations to perform.
       \param w The relaxation factor.
     */
    SeqJac (const M& A, int n, real_field_type w)
      : _A_(A), _n(n), _w(w)
    {
      CheckIfDiagonalPresent<M,l>::check(_A_);
    }

    /*!
       \brief Constructor.

       \param A The assembled linear operator to use.
       \param configuration ParameterTree containing preconditioner parameters.

       ParameterTree Key | Meaning
       ------------------|------------
       iterations        | The number of iterations to perform. default=1
       relaxation        | The relaxation factor. default=1.0

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    SeqJac (const std::shared_ptr<const AssembledLinearOperator<M,X,Y>>& A, const ParameterTree& configuration)
      : SeqJac(A->getmat(), configuration)
    {}

    /*!
       \brief Constructor.

       \param A The matrix to operate on.
       \param configuration ParameterTree containing preconditioner parameters.

       ParameterTree Key | Meaning
       ------------------|------------
       iterations        | The number of iterations to perform. default=1
       relaxation        | The relaxation factor. default=1.0

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    SeqJac (const M& A, const ParameterTree& configuration)
      : SeqJac(A, configuration.get<int>("iterations",1), configuration.get<real_field_type>("relaxation",1.0))
    {}

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre ([[maybe_unused]] X& x, [[maybe_unused]] Y& b)
    {}

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
    virtual void post ([[maybe_unused]] X& x)
    {}

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::sequential;
    }

  private:
    //! \brief The matrix we operate on.
    const M& _A_;
    //! \brief The number of steps to perform during apply.
    int _n;
    //! \brief The relaxation parameter to use.
    real_field_type _w;
  };
  DUNE_REGISTER_PRECONDITIONER("jac", defaultPreconditionerBlockLevelCreator<Dune::SeqJac>());



  /*!
     \brief Sequential ILU preconditioner.

     Wraps the naked ISTL generic ILU preconditioner into the solver framework.

     \tparam M The matrix type to operate on
     \tparam X Type of the update
     \tparam Y Type of the defect
     \tparam l Ignored. Just there to have the same number of template arguments
     as other preconditioners.
   */
  template<class M, class X, class Y, int l=1>
  class SeqILU : public Preconditioner<X,Y> {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef typename std::remove_const<M>::type matrix_type;
    //! block type of matrix
    typedef typename matrix_type :: block_type block_type;
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;

    //! \brief The field type of the preconditioner.
    typedef typename X::field_type field_type;

    //! \brief scalar type underlying the field_type
    typedef Simd::Scalar<field_type> scalar_field_type;
    //! \brief real scalar type underlying the field_type
    typedef typename FieldTraits<scalar_field_type>::real_type real_field_type;

    //! \brief type of ILU storage
    typedef typename ILU::CRS< block_type , typename M::allocator_type> CRS;

    /*! \brief Constructor.

       Constructor invoking ILU(0) gets all parameters to operate the prec.
       \param A The matrix to operate on.
       \param w The relaxation factor.
       \param resort true if a resort of the computed ILU for improved performance should be done.
     */
    SeqILU (const M& A, real_field_type w, const bool resort = false )
      : SeqILU( A, 0, w, resort ) // construct ILU(0)
    {
    }

    /*!
      \brief Constructor.

      \param A The assembled linear operator to use.
      \param configuration ParameterTree containing preconditioner parameters.

      ParameterTree Key | Meaning
      ------------------|------------
      n                 | The order of the ILU decomposition. default=0
      relaxation        | The relaxation factor. default=1.0
      resort            | True if a resort of the computed ILU for improved performance should be done. default=false

      See \ref ISTL_Factory for the ParameterTree layout and examples.
    */
    SeqILU (const std::shared_ptr<const AssembledLinearOperator<M,X,Y>>& A, const ParameterTree& configuration)
      : SeqILU(A->getmat(), configuration)
    {}

    /*!
       \brief Constructor.

       \param A The matrix to operate on.
       \param configuration ParameterTree containing preconditioner parameters.

       ParameterTree Key | Meaning
       ------------------|------------
      n                 | The order of the ILU decomposition. default=0
      relaxation        | The relaxation factor. default=1.0
      resort            | True if a resort of the computed ILU for improved performance should be done. default=false

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    SeqILU(const M& A, const ParameterTree& config)
      : SeqILU(A, config.get("n", 0),
               config.get<real_field_type>("relaxation", 1.0),
               config.get("resort", false))
    {}

   /*! \brief Constructor.

       Constructor invoking ILU(n).
       \param A The matrix to operate on.
       \param n The order of the ILU decomposition.
       \param w The relaxation factor.
       \param resort true if a resort of the computed ILU for improved performance should be done.
     */
    SeqILU (const M& A, int n, real_field_type w, const bool resort = false )
      : ILU_(),
        lower_(),
        upper_(),
        inv_(),
        w_(w),
        wNotIdentity_([w]{using std::abs; return abs(w - real_field_type(1)) > 1e-15;}() )
    {
      if( n == 0 )
      {
        // copy A
        ILU_.reset( new matrix_type( A ) );
        // create ILU(0) decomposition
        ILU::blockILU0Decomposition( *ILU_ );
      }
      else
      {
        // create matrix in build mode
        ILU_.reset( new matrix_type(  A.N(), A.M(), matrix_type::row_wise) );
        // create ILU(n) decomposition
        ILU::blockILUDecomposition( A, n, *ILU_ );
      }

      if( resort )
      {
        // store ILU in simple CRS format
        ILU::convertToCRS( *ILU_, lower_, upper_, inv_ );
        ILU_.reset();
      }
    }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre ([[maybe_unused]] X& x, [[maybe_unused]] Y& b)
    {}

    /*!
       \brief Apply the preconditioner.

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      if( ILU_ )
      {
        ILU::blockILUBacksolve( *ILU_, v, d);
      }
      else
      {
        ILU::blockILUBacksolve(lower_, upper_, inv_, v, d);
      }

      if( wNotIdentity_ )
      {
        v *= w_;
      }
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post ([[maybe_unused]] X& x)
    {}

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::sequential;
    }

  protected:
    //! \brief The ILU(n) decomposition of the matrix. As storage a BCRSMatrix is used.
    std::unique_ptr< matrix_type > ILU_;

    //! \brief The ILU(n) decomposition of the matrix. As storage a CRS structure is used.
    CRS lower_;
    CRS upper_;
    std::vector< block_type, typename matrix_type::allocator_type > inv_;

    //! \brief The relaxation factor to use.
    const real_field_type w_;
    //! \brief true if w != 1.0
    const bool wNotIdentity_;
  };
  DUNE_REGISTER_PRECONDITIONER("ilu", defaultPreconditionerBlockLevelCreator<Dune::SeqILU>());


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
    //! \brief scalar type underlying the field_type
    typedef Simd::Scalar<field_type> scalar_field_type;
    //! \brief real scalar type underlying the field_type
    typedef typename FieldTraits<scalar_field_type>::real_type real_field_type;

    /*! \brief Constructor.

       Constructor gets all parameters to operate the prec.
       \param w The relaxation factor.
     */
    Richardson (real_field_type w=1.0) :
      _w(w)
    {}

    /*!
       \brief Constructor.

       \param configuration ParameterTree containing preconditioner parameters.

       ParameterTree Key | Meaning
       ------------------|------------
       relaxation        | The relaxation factor. default=1.0

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    Richardson (const ParameterTree& configuration)
      : Richardson(configuration.get<real_field_type>("relaxation", 1.0))
    {}

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre ([[maybe_unused]] X& x, [[maybe_unused]] Y& b)
    {}

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
    virtual void post ([[maybe_unused]] X& x)
    {}

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::sequential;
    }

  private:
    //! \brief The relaxation factor to use.
    real_field_type _w;
  };
  DUNE_REGISTER_PRECONDITIONER("richardson", [](auto tl, const auto& /* mat */, const ParameterTree& config){
                                               using D = typename Dune::TypeListElement<1, decltype(tl)>::type;
                                               using R = typename Dune::TypeListElement<2, decltype(tl)>::type;
                                               return std::make_shared<Richardson<D,R>>(config);
                                             });


  /**
   * \brief   sequential ILDL preconditioner
   * \author  Martin Nolte
   *
   * Wraps the naked ISTL generic ILDL preconditioner into the solver framework.
   *
   * \tparam  M  type of matrix to operate on
   * \tparam  X  type of update
   * \tparam  Y  type of defect
   **/
  template< class M, class X, class Y >
  class SeqILDL
    : public Preconditioner< X, Y >
  {
    typedef SeqILDL< M, X, Y > This;
    typedef Preconditioner< X, Y > Base;

  public:
    /** \brief type of matrix the preconditioner is for **/
    typedef std::remove_const_t< M > matrix_type;
    /** \brief domain type of the preconditioner **/
    typedef X domain_type;
    /** \brief range type of the preconditioner **/
    typedef Y range_type;
    /** \brief field type of the preconditioner **/
    typedef typename X::field_type field_type;
    //! \brief scalar type underlying the field_type
    typedef Simd::Scalar<field_type> scalar_field_type;
    //! \brief real scalar type underlying the field_type
    typedef typename FieldTraits<scalar_field_type>::real_type real_field_type;

    /*!
       \brief Constructor.

       \param A The linear operator to use.
       \param configuration ParameterTree containing preconditioner parameters.

       ParameterTree Key | Meaning
       ------------------|------------
       relaxation        | relaxation factor

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    SeqILDL (const std::shared_ptr<const AssembledLinearOperator<M,X,Y>>& A, const ParameterTree& configuration)
      : SeqILDL(A->getmat(), configuration)
    {}

    /*!
       \brief Constructor.

       \param A The matrix to operate on.
       \param configuration ParameterTree containing preconditioner parameters.

       ParameterTree Key | Meaning
       ------------------|------------
       relaxation        | relaxation factor. default=1.0

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    SeqILDL(const matrix_type& A, const ParameterTree& config)
      : SeqILDL(A, config.get<real_field_type>("relaxation", 1.0))
    {}

    /**
     * \brief constructor
     *
     * The constructor copies the matrix A and computes its ILDL decomposition.
     *
     * \param[in]  A      matrix to operate on
     * \param[in]  relax  relaxation factor
     **/
    explicit SeqILDL ( const matrix_type &A, real_field_type relax = real_field_type( 1 ) )
      : decomposition_( A.N(), A.M(), matrix_type::random ),
        relax_( relax )
    {
      // setup row sizes for lower triangular matrix
      for( auto i = A.begin(), iend = A.end(); i != iend; ++i )
      {
        const auto &A_i = *i;
        const auto ij = A_i.find( i.index() );
        if( ij != A_i.end() )
          decomposition_.setrowsize( i.index(), ij.offset()+1 );
        else
          DUNE_THROW( ISTLError, "diagonal entry missing" );
      }
      decomposition_.endrowsizes();

      // setup row indices for lower triangular matrix
      for( auto i = A.begin(), iend = A.end(); i != iend; ++i )
      {
        const auto &A_i = *i;
        for( auto ij = A_i.begin(); ij.index() < i.index() ; ++ij )
          decomposition_.addindex( i.index(), ij.index() );
        decomposition_.addindex( i.index(), i.index() );
      }
      decomposition_.endindices();

      // copy values of lower triangular matrix
      auto i = A.begin();
      for( auto row = decomposition_.begin(), rowend = decomposition_.end(); row != rowend; ++row, ++i )
      {
        auto ij = i->begin();
        for( auto col = row->begin(), colend = row->end(); col != colend; ++col, ++ij )
          *col = *ij;
      }

      // perform ILDL decomposition
      bildl_decompose( decomposition_ );
    }

    /** \copydoc Preconditioner::pre(X&,Y&) **/
    void pre ([[maybe_unused]] X &x, [[maybe_unused]] Y &b) override
    {}

    /** \copydoc Preconditioner::apply(X&,const Y&) **/
    void apply ( X &v, const Y &d ) override
    {
      bildl_backsolve( decomposition_, v, d, true );
      v *= relax_;
    }

    /** \copydoc Preconditioner::post(X&) **/
    void post ([[maybe_unused]] X &x) override
    {}

    /** \copydoc Preconditioner::category() **/
    SolverCategory::Category category () const override { return SolverCategory::sequential; }

  private:
    matrix_type decomposition_;
    real_field_type relax_;
  };
  DUNE_REGISTER_PRECONDITIONER("ildl", defaultPreconditionerCreator<Dune::SeqILDL>());

  /** @} end documentation */

} // end namespace


#endif
