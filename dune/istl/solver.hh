// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVER_HH
#define DUNE_ISTL_SOLVER_HH

#include <iomanip>
#include <ostream>
#include <string>
#include <functional>

#include <dune/common/exceptions.hh>
#include <dune/common/shared_ptr.hh>
#include <dune/common/simd/io.hh>
#include <dune/common/simd/simd.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/timer.hh>

#include "solvertype.hh"
#include "preconditioner.hh"
#include "operators.hh"
#include "scalarproducts.hh"

namespace Dune
{
/**
 * @addtogroup ISTL_Solvers
 * @{
 */
/** \file

      \brief   Define general, extensible interface for inverse operators.

      Implementation here covers only inversion of linear operators,
      but the implementation might be used for nonlinear operators
      as well.
   */
  /**
      \brief Statistics about the application of an inverse operator

      The return value of an application of the inverse
      operator delivers some important information about
      the iteration.
   */
  struct InverseOperatorResult
  {
    /** \brief Default constructor */
    InverseOperatorResult ()
    {
      clear();
    }

    /** \brief Resets all data */
    void clear ()
    {
      iterations = 0;
      reduction = 0;
      converged = false;
      conv_rate = 1;
      elapsed = 0;
      condition_estimate = -1;
    }

    /** \brief Number of iterations */
    int iterations;

    /** \brief Reduction achieved: \f$ \|b-A(x^n)\|/\|b-A(x^0)\|\f$ */
    double reduction;

    /** \brief True if convergence criterion has been met */
    bool converged;

    /** \brief Convergence rate (average reduction per step) */
    double conv_rate;

    /** \brief Estimate of condition number */
    double condition_estimate = -1;

    /** \brief Elapsed time in seconds */
    double elapsed;
  };


  //=====================================================================
  /*!
     \brief Abstract base class for all solvers.

     An InverseOperator computes the solution of \f$ A(x)=b\f$ where
     \f$ A : X \to Y \f$ is an operator.
     Note that the solver "knows" which operator
     to invert and which preconditioner to apply (if any). The
     user is only interested in inverting the operator.
     InverseOperator might be a Newton scheme, a Krylov subspace method,
     or a direct solver or just anything.
   */
  template<class X, class Y>
  class InverseOperator {
  public:
    //! \brief Type of the domain of the operator to be inverted.
    typedef X domain_type;

    //! \brief Type of the range of the operator to be inverted.
    typedef Y range_type;

    /** \brief The field type of the operator. */
    typedef typename X::field_type field_type;

    //! \brief The real type of the field type (is the same if using real numbers, but differs for std::complex)
    typedef typename FieldTraits<field_type>::real_type real_type;

    //! \brief scalar type underlying the field_type
    typedef Simd::Scalar<real_type> scalar_real_type;

    /**
        \brief Apply inverse operator,

        \warning Note: right hand side b may be overwritten!

        \param x The left hand side to store the result in.
        \param b The right hand side
        \param res Object to store the statistics about applying the operator.

        \throw SolverAbort When the solver detects a problem and cannot
                           continue
     */
    virtual void apply (X& x, Y& b, InverseOperatorResult& res) = 0;

    /*!
       \brief apply inverse operator, with given convergence criteria.

       \warning Right hand side b may be overwritten!

       \param x The left hand side to store the result in.
       \param b The right hand side
       \param reduction The minimum defect reduction to achieve.
       \param res Object to store the statistics about applying the operator.

       \throw SolverAbort When the solver detects a problem and cannot
                          continue
     */
    virtual void apply (X& x, Y& b, double reduction, InverseOperatorResult& res) = 0;

    //! Category of the solver (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
#ifdef DUNE_ISTL_SUPPORT_OLD_CATEGORY_INTERFACE
    {
      DUNE_THROW(Dune::Exception,"It is necessary to implement the category method in a derived classes, in the future this method will pure virtual.");
    }
#else
    = 0;
#endif

    //! \brief Destructor
    virtual ~InverseOperator () {}

  protected:
    // spacing values
    enum { iterationSpacing = 5 , normSpacing = 16 };

    //! helper function for printing header of solver output
    void printHeader(std::ostream& s) const
    {
      s << std::setw(iterationSpacing)  << " Iter";
      s << std::setw(normSpacing) << "Defect";
      s << std::setw(normSpacing) << "Rate" << std::endl;
    }

    //! helper function for printing solver output
    template <typename CountType, typename DataType>
    void printOutput(std::ostream& s,
                     const CountType& iter,
                     const DataType& norm,
                     const DataType& norm_old) const
    {
      const DataType rate = norm/norm_old;
      s << std::setw(iterationSpacing)  << iter << " ";
      s << std::setw(normSpacing) << Simd::io(norm) << " ";
      s << std::setw(normSpacing) << Simd::io(rate) << std::endl;
    }

    //! helper function for printing solver output
    template <typename CountType, typename DataType>
    void printOutput(std::ostream& s,
                     const CountType& iter,
                     const DataType& norm) const
    {
      s << std::setw(iterationSpacing)  << iter << " ";
      s << std::setw(normSpacing) << Simd::io(norm) << std::endl;
    }
  };

  /*!
     \brief Base class for all implementations of iterative solvers

     This class provides all storage, which is needed by the usual
     iterative solvers. In additional it provides all the necessary
     constructors, which are then only imported in the actual solver
     implementation.
   */
  template<class X, class Y>
  class IterativeSolver : public InverseOperator<X,Y>{
  public:
    using typename InverseOperator<X,Y>::domain_type;
    using typename InverseOperator<X,Y>::range_type;
    using typename InverseOperator<X,Y>::field_type;
    using typename InverseOperator<X,Y>::real_type;
    using typename InverseOperator<X,Y>::scalar_real_type;

    /*!
       \brief General constructor to initialize an iterative solver.

       \param op The operator we solve.
       \param prec The preconditioner to apply in each iteration of the loop.
       Has to inherit from Preconditioner.
       \param reduction The relative defect reduction to achieve when applying
       the operator.
       \param maxit The maximum number of iteration steps allowed when applying
       the operator.
       \param verbose The verbosity level.

       Verbose levels are:
       <ul>
       <li> 0 : print nothing </li>
       <li> 1 : print initial and final defect and statistics </li>
       <li> 2 : print line for each iteration </li>
       </ul>
     */
    IterativeSolver (const LinearOperator<X,Y>& op, Preconditioner<X,Y>& prec, scalar_real_type reduction, int maxit, int verbose) :
      _op(stackobject_to_shared_ptr(op)),
      _prec(stackobject_to_shared_ptr(prec)),
      _sp(new SeqScalarProduct<X>),
      _reduction(reduction), _maxit(maxit), _verbose(verbose), _category(SolverCategory::sequential)
    {
      if(SolverCategory::category(op) != SolverCategory::sequential)
        DUNE_THROW(InvalidSolverCategory, "LinearOperator has to be sequential!");
      if(SolverCategory::category(prec) != SolverCategory::sequential)
        DUNE_THROW(InvalidSolverCategory, "Preconditioner has to be sequential!");
    }

    /**
        \brief General constructor to initialize an iterative solver

        \param op The operator we solve.
        \param sp The scalar product to use, e. g. SeqScalarproduct.
        \param prec The preconditioner to apply in each iteration of the loop.
        Has to inherit from Preconditioner.
        \param reduction The relative defect reduction to achieve when applying
        the operator.
        \param maxit The maximum number of iteration steps allowed when applying
        the operator.
        \param verbose The verbosity level.

        Verbose levels are:
        <ul>
        <li> 0 : print nothing </li>
        <li> 1 : print initial and final defect and statistics </li>
        <li> 2 : print line for each iteration </li>
        </ul>
     */
    IterativeSolver (const LinearOperator<X,Y>& op, const ScalarProduct<X>& sp, Preconditioner<X,Y>& prec,
      scalar_real_type reduction, int maxit, int verbose) :
      _op(stackobject_to_shared_ptr(op)),
      _prec(stackobject_to_shared_ptr(prec)),
      _sp(stackobject_to_shared_ptr(sp)),
      _reduction(reduction), _maxit(maxit), _verbose(verbose), _category(SolverCategory::category(op))
    {
      if(SolverCategory::category(op) != SolverCategory::category(prec))
        DUNE_THROW(InvalidSolverCategory, "LinearOperator and Preconditioner must have the same SolverCategory!");
      if(SolverCategory::category(op) != SolverCategory::category(sp))
        DUNE_THROW(InvalidSolverCategory, "LinearOperator and ScalarProduct must have the same SolverCategory!");
    }

        /*!
       \brief Constructor.

       \param op The operator we solve
       \param prec The preconditioner to apply in each iteration of the loop.
       \param configuration ParameterTree containing iterative solver parameters.

       ParameterTree Key | Meaning
       ------------------|------------
       reduction         | The relative defect reduction to achieve when applying the operator
       maxit             | The maximum number of iteration steps allowed when applying the operator
       verbose           | The verbosity level

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    IterativeSolver (std::shared_ptr<const LinearOperator<X,Y> > op, std::shared_ptr<Preconditioner<X,X> > prec, const ParameterTree& configuration) :
      IterativeSolver(op,std::make_shared<SeqScalarProduct<X>>(),prec,
        configuration.get<real_type>("reduction"),
        configuration.get<int>("maxit"),
        configuration.get<int>("verbose"))
    {}

    /*!
       \brief Constructor.

       \param op The operator we solve
       \param sp The scalar product to use, e. g. SeqScalarproduct.
       \param prec The preconditioner to apply in each iteration of the loop.
       \param configuration ParameterTree containing iterative solver parameters.

       ParameterTree Key | Meaning
       ------------------|------------
       reduction         | The relative defect reduction to achieve when applying the operator
       maxit             | The maximum number of iteration steps allowed when applying the operator
       verbose           | The verbosity level

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    IterativeSolver (std::shared_ptr<const LinearOperator<X,Y> > op, std::shared_ptr<const ScalarProduct<X> > sp, std::shared_ptr<Preconditioner<X,X> > prec, const ParameterTree& configuration) :
      IterativeSolver(op,sp,prec,
        configuration.get<scalar_real_type>("reduction"),
        configuration.get<int>("maxit"),
        configuration.get<int>("verbose"))
    {}

    /**
        \brief General constructor to initialize an iterative solver

        \param op The operator we solve.
        \param sp The scalar product to use, e. g. SeqScalarproduct.
        \param prec The preconditioner to apply in each iteration of the loop.
        Has to inherit from Preconditioner.
        \param reduction The relative defect reduction to achieve when applying
        the operator.
        \param maxit The maximum number of iteration steps allowed when applying
        the operator.
        \param verbose The verbosity level.

        Verbose levels are:
        <ul>
        <li> 0 : print nothing </li>
        <li> 1 : print initial and final defect and statistics </li>
        <li> 2 : print line for each iteration </li>
        </ul>
     */
    IterativeSolver (std::shared_ptr<const LinearOperator<X,Y>> op,
                     std::shared_ptr<const ScalarProduct<X>> sp,
                     std::shared_ptr<Preconditioner<X,Y>> prec,
                     scalar_real_type reduction, int maxit, int verbose) :
      _op(op),
      _prec(prec),
      _sp(sp),
      _reduction(reduction), _maxit(maxit), _verbose(verbose),
      _category(SolverCategory::category(*op))
    {
      if(SolverCategory::category(*op) != SolverCategory::category(*prec))
        DUNE_THROW(InvalidSolverCategory, "LinearOperator and Preconditioner must have the same SolverCategory!");
      if(SolverCategory::category(*op) != SolverCategory::category(*sp))
        DUNE_THROW(InvalidSolverCategory, "LinearOperator and ScalarProduct must have the same SolverCategory!");
    }

    // #warning actually we want to have this as the default and just implement the second one
    // //! \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)
    // virtual void apply (X& x, Y& b, InverseOperatorResult& res)
    // {
    //   apply(x,b,_reduction,res);
    // }

#ifndef DOXYGEN
    // make sure the three-argument apply from the base class does not get shadowed
    // by the redefined four-argument version below
    using InverseOperator<X,Y>::apply;
#endif

    /*!
       \brief Apply inverse operator with given reduction factor.

       \copydoc InverseOperator::apply(X&,Y&,double,InverseOperatorResult&)
     */
    virtual void apply (X& x, X& b, double reduction, InverseOperatorResult& res)
    {
      scalar_real_type saved_reduction = _reduction;
      _reduction = reduction;
      this->apply(x,b,res);
      _reduction = saved_reduction;
    }

    //! Category of the solver (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return _category;
    }

    std::string name() const{
      std::string name = className(*this);
      return name.substr(0, name.find("<"));
    }

      /*!
     \brief Class for controlling iterative methods

     This class provides building blocks for a iterative method. It does all
     things that have to do with output, residual checking (NaN, infinite,
     convergence) and sets also the fields of InverseOperatorResult.

     Instances of this class are meant to create with
     IterativeSolver::startIteration and stored as a local variable in the apply
     method. If the scope of the apply method is left the destructor of this
     class sets all the solver statistics in the InverseOperatorResult and
     prints the final output.

     During the iteration in every step Iteration::step should be called with
     the current iteration count and norm of the residual. It returns true if
     convergence is achieved.
   */
    template<class CountType = unsigned int>
    class Iteration {
    public:
      Iteration(const IterativeSolver& parent, InverseOperatorResult& res)
        : _i(0)
        , _res(res)
        , _parent(parent)
        , _valid(true)
      {
        res.clear();
        if(_parent._verbose>0){
          std::cout << "=== " << parent.name() << std::endl;
          if(_parent._verbose > 1)
            _parent.printHeader(std::cout);
        }
      }

      Iteration(const Iteration&) = delete;
      Iteration(Iteration&& other)
        : _def0(other._def0)
        , _def(other._def)
        , _i(other._i)
        , _watch(other._watch)
        , _res(other._res)
        , _parent(other._parent)
        , _valid(other._valid)
      {
        other._valid = false;
      }

      ~Iteration(){
        if(_valid)
          finalize();
      }

      /*! \brief registers the iteration step, checks for invalid defect norm
          and convergence.

        \param i The current iteration count
        \param def The current norm of the defect

        \return true is convergence is achieved

        \throw SolverAbort when `def` contains inf or NaN
       */
      bool step(CountType i, real_type def){
        if (!Simd::allTrue(isFinite(def))) // check for inf or NaN
        {
          if (_parent._verbose>0)
            std::cout << "=== " << _parent.name() << ": abort due to infinite or NaN defect"
                      << std::endl;
          DUNE_THROW(SolverAbort,
                     _parent.name() << ": defect=" << Simd::io(def)
                     << " is infinite or NaN");
        }
        if(i == 0)
          _def0 = def;
        if(_parent._verbose > 1){
          if(i!=0)
            _parent.printOutput(std::cout,i,def,_def);
          else
            _parent.printOutput(std::cout,i,def);
        }
        _def = def;
        _i = i;
        _res.converged = (Simd::allTrue(def<_def0*_parent._reduction || def<real_type(1E-30)));    // convergence check
        return _res.converged;
      }

    protected:
      void finalize(){
        _res.converged = (Simd::allTrue(_def<_def0*_parent._reduction || _def<real_type(1E-30)));
        _res.iterations = _i;
        _res.reduction = static_cast<double>(Simd::max(_def/_def0));
        _res.conv_rate  = pow(_res.reduction,1.0/_i);
        _res.elapsed = _watch.elapsed();
        if (_parent._verbose>0)                 // final print
          {
            std::cout << "=== rate=" << _res.conv_rate
                      << ", T=" << _res.elapsed
                      << ", TIT=" << _res.elapsed/_res.iterations
                      << ", IT=" << _res.iterations << std::endl;
          }
      }

      real_type _def0 = 0.0, _def = 0.0;
      CountType _i;
      Timer _watch;
      InverseOperatorResult& _res;
      const IterativeSolver& _parent;
      bool _valid;
    };

  protected:
    std::shared_ptr<const LinearOperator<X,Y>> _op;
    std::shared_ptr<Preconditioner<X,Y>> _prec;
    std::shared_ptr<const ScalarProduct<X>> _sp;
    scalar_real_type _reduction;
    int _maxit;
    int _verbose;
    SolverCategory::Category _category;
  };

  /**
   * \brief Helper class for notifying a DUNE-ISTL linear solver about
   *        a change of the iteration matrix object in a unified way,
   *        i.e. independent from the solver's type (direct/iterative).
   *
   * \author Sebastian Westerheide.
   */
  template <typename ISTLLinearSolver, typename BCRSMatrix>
  class SolverHelper
  {
  public:
    static void setMatrix (ISTLLinearSolver& solver,
                           const BCRSMatrix& matrix)
    {
      static const bool is_direct_solver
        = Dune::IsDirectSolver<ISTLLinearSolver>::value;
      SolverHelper<ISTLLinearSolver,BCRSMatrix>::
        Implementation<is_direct_solver>::setMatrix(solver,matrix);
    }

  protected:
    /**
     * \brief Implementation that works together with iterative ISTL
     *        solvers, e.g. Dune::CGSolver or Dune::BiCGSTABSolver.
     */
    template <bool is_direct_solver, typename Dummy = void>
    struct Implementation
    {
      static void setMatrix (ISTLLinearSolver&,
                             const BCRSMatrix&)
      {}
    };

    /**
     * \brief Implementation that works together with direct ISTL
     *        solvers, e.g. Dune::SuperLU or Dune::UMFPack.
     */
    template <typename Dummy>
    struct Implementation<true,Dummy>
    {
      static void setMatrix (ISTLLinearSolver& solver,
                             const BCRSMatrix& matrix)
      {
        solver.setMatrix(matrix);
      }
    };
  };

/**
 * @}
 */
}

#endif
