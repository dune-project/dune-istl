// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVER_HH
#define DUNE_ISTL_SOLVER_HH

#include <iomanip>
#include <ostream>

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

/**
 * @}
 */
}

#endif
