// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVER_HH
#define DUNE_ISTL_SOLVER_HH

#include <iomanip>
#include <ostream>

#include <dune/common/exceptions.hh>
#include <dune/common/shared_ptr.hh>
#include <dune/common/parametertree.hh>

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
    }

    /** \brief Number of iterations */
    int iterations;

    /** \brief Reduction achieved: \f$ \|b-A(x^n)\|/\|b-A(x^0)\|\f$ */
    double reduction;

    /** \brief True if convergence criterion has been met */
    bool converged;

    /** \brief Convergence rate (average reduction per step) */
    double conv_rate;

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
      s << std::setw(normSpacing) << norm << " ";
      s << std::setw(normSpacing) << rate << std::endl;
    }

    //! helper function for printing solver output
    template <typename CountType, typename DataType>
    void printOutput(std::ostream& s,
                     const CountType& iter,
                     const DataType& norm) const
    {
      s << std::setw(iterationSpacing)  << iter << " ";
      s << std::setw(normSpacing) << norm << std::endl;
    }
  };

  template<class X, class Y>
  class IterativeSolver : public InverseOperator<X,Y>{
  public:
    using typename InverseOperator<X,Y>::domain_type;
    using typename InverseOperator<X,Y>::range_type;
    using typename InverseOperator<X,Y>::field_type;
    using typename InverseOperator<X,Y>::real_type;

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
    IterativeSolver (LinearOperator<X,Y>& op, Preconditioner<X,Y>& prec, real_type reduction, int maxit, int verbose) :
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
    IterativeSolver (LinearOperator<X,Y>& op, ScalarProduct<X>& sp, Preconditioner<X,Y>& prec,
      real_type reduction, int maxit, int verbose) :
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

    IterativeSolver (std::shared_ptr<LinearOperator<X,Y> > op, std::shared_ptr<Preconditioner<X,X> > prec, const ParameterTree& configuration) :
      _op(op),
      _prec(prec),
      _sp(new SeqScalarProduct<X>),
      _reduction(configuration.get<real_type>("reduction")),
      _maxit(configuration.get<int>("maxit")),
      _verbose(configuration.get<int>("verbose")),
      _category(SolverCategory::category(*op))
    {
      if(SolverCategory::category(*op) != SolverCategory::sequential)
        DUNE_THROW(InvalidSolverCategory, "LinearOperator has to be sequential!");
      if(SolverCategory::category(*prec) != SolverCategory::sequential)
        DUNE_THROW(InvalidSolverCategory, "Preconditioner has to be sequential!");
    }

    IterativeSolver (std::shared_ptr<LinearOperator<X,Y> > op, std::shared_ptr<ScalarProduct<X> > sp, std::shared_ptr<Preconditioner<X,X> > prec, const ParameterTree& configuration) :
      _op(op),
      _prec(prec),
      _sp(sp),
      _reduction(configuration.get<real_type>("reduction")),
      _maxit(configuration.get<int>("maxit")),
      _verbose(configuration.get<int>("verbose")),
      _category(SolverCategory::category(*op))
    {
      if(SolverCategory::category(*op) != SolverCategory::category(*prec))
        DUNE_THROW(InvalidSolverCategory, "LinearOperator and Preconditioner must have the same SolverCategory!");
      if(SolverCategory::category(*op) != SolverCategory::category(*sp))
        DUNE_THROW(InvalidSolverCategory, "LinearOperator and ScalarProduct must have the same SolverCategory!");
    }

#warning actually we want to have this as the default and just implement the second one
    // //! \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)
    // virtual void apply (X& x, Y& b, InverseOperatorResult& res)
    // {
    //   apply(x,b,_reduction,res);
    // }

    /*!
       \brief Apply inverse operator with given reduction factor.

       \copydoc InverseOperator::apply(X&,Y&,double,InverseOperatorResult&)
     */
    virtual void apply (X& x, X& b, double reduction, InverseOperatorResult& res)
    {
      real_type saved_reduction = _reduction;
      _reduction = reduction;
      static_cast<InverseOperator<X,Y>*>(this)->apply(x,b,res);
      _reduction = saved_reduction;
    }

    //! Category of the solver (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return _category;
    }

  protected:
    std::shared_ptr<LinearOperator<X,Y>> _op;
    std::shared_ptr<Preconditioner<X,Y>> _prec;
    std::shared_ptr<ScalarProduct<X>> _sp;
    real_type _reduction;
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
