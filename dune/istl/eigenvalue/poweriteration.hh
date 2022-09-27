// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_EIGENVALUE_POWERITERATION_HH
#define DUNE_ISTL_EIGENVALUE_POWERITERATION_HH

#include <cstddef>  // provides std::size_t
#include <cmath>    // provides std::sqrt, std::abs

#include <type_traits>  // provides std::is_same
#include <iostream>     // provides std::cout, std::endl
#include <limits>       // provides std::numeric_limits
#include <ios>          // provides std::left, std::ios::left
#include <iomanip>      // provides std::setw, std::resetiosflags
#include <memory>       // provides std::unique_ptr
#include <string>       // provides std::string

#include <dune/common/exceptions.hh>  // provides DUNE_THROW(...)

#include <dune/istl/blocklevel.hh>      // provides Dune::blockLevel
#include <dune/istl/operators.hh>       // provides Dune::LinearOperator
#include <dune/istl/solvercategory.hh>  // provides Dune::SolverCategory::sequential
#include <dune/istl/solvertype.hh>      // provides Dune::IsDirectSolver
#include <dune/istl/operators.hh>       // provides Dune::MatrixAdapter
#include <dune/istl/istlexception.hh>   // provides Dune::ISTLError
#include <dune/istl/io.hh>              // provides Dune::printvector(...)
#include <dune/istl/solvers.hh>         // provides Dune::InverseOperatorResult

namespace Dune
{

  /** @addtogroup ISTL_Eigenvalue
    @{
  */

  namespace Impl {
    /**
     * \brief A linear operator scaling vectors by a scalar value.
     *        The scalar value can be changed as it is given in a
     *        form decomposed into an immutable and a mutable part.
     *
     * \author Sebastian Westerheide.
     */
    template <class X, class Y = X>
    class ScalingLinearOperator : public Dune::LinearOperator<X,Y>
    {
    public:
      typedef X domain_type;
      typedef Y range_type;
      typedef typename X::field_type field_type;

      ScalingLinearOperator (field_type immutable_scaling,
        const field_type& mutable_scaling)
        : immutable_scaling_(immutable_scaling),
          mutable_scaling_(mutable_scaling)
      {}

      virtual void apply (const X& x, Y& y) const
      {
        y = x;
        y *= immutable_scaling_*mutable_scaling_;
      }

      virtual void applyscaleadd (field_type alpha, const X& x, Y& y) const
      {
        X temp(x);
        temp *= immutable_scaling_*mutable_scaling_;
        y.axpy(alpha,temp);
      }

      //! Category of the linear operator (see SolverCategory::Category)
      virtual SolverCategory::Category category() const
      {
        return SolverCategory::sequential;
      }

    protected:
      const field_type immutable_scaling_;
      const field_type& mutable_scaling_;
    };


    /**
     * \brief A linear operator representing the sum of two linear operators.
     *
     * \tparam OP1 Type of the first linear operator.
     * \tparam OP2 Type of the second linear operator.
     *
     * \author Sebastian Westerheide.
     */
    template <class OP1, class OP2>
    class LinearOperatorSum
      : public Dune::LinearOperator<typename OP1::domain_type,
                                    typename OP1::range_type>
    {
    public:
      typedef typename OP1::domain_type domain_type;
      typedef typename OP1::range_type range_type;
      typedef typename domain_type::field_type field_type;

      LinearOperatorSum (const OP1& op1, const OP2& op2)
        : op1_(op1), op2_(op2)
      {
        static_assert(std::is_same<typename OP2::domain_type,domain_type>::value,
          "Domain type of both operators doesn't match!");
        static_assert(std::is_same<typename OP2::range_type,range_type>::value,
          "Range type of both operators doesn't match!");
      }

      virtual void apply (const domain_type& x, range_type& y) const
      {
        op1_.apply(x,y);
        op2_.applyscaleadd(1.0,x,y);
      }

      virtual void applyscaleadd (field_type alpha,
        const domain_type& x, range_type& y) const
      {
        range_type temp(y);
        op1_.apply(x,temp);
        op2_.applyscaleadd(1.0,x,temp);
        y.axpy(alpha,temp);
      }

      //! Category of the linear operator (see SolverCategory::Category)
      virtual SolverCategory::Category category() const
      {
        return SolverCategory::sequential;
      }

    protected:
      const OP1& op1_;
      const OP2& op2_;
    };
  } // end namespace Impl

  /**
   * \brief Iterative eigenvalue algorithms based on power iteration.
   *
   * Given a square matrix whose eigenvalues shall be considered, this class
   * template provides methods for performing the power iteration algorithm,
   * the inverse iteration algorithm, the inverse iteration with shift algorithm,
   * the Rayleigh quotient iteration algorithm and the TLIME iteration algorithm.
   *
   * \note Note that all algorithms except the power iteration algorithm require
   *       matrix inversion via a linear solver. When using an iterative linear
   *       solver, the algorithms become inexact "inner-outer" iterative methods.
   *       It is known that the number of inner solver iterations can increase
   *       steadily as the outer eigenvalue iteration proceeds. In this case, you
   *       should consider using a "tuned preconditioner", see e.g. [Freitag and
   *       Spence, 2008].
   *
   * \note In the current implementation, preconditioners like Dune::SeqILUn
   *       which are based on matrix decomposition act on the initial iteration
   *       matrix in each iteration, even for methods like the Rayleigh quotient
   *       algorithm in which the iteration matrix (m_ - mu_*I) may change in
   *       each iteration. This is due to the fact that those preconditioners
   *       currently don't support to be notified about a change of the matrix
   *       object.
   *
   * \todo The current implementation is limited to DUNE-ISTL BCRSMatrix types
   *       with blocklevel 2. An extension to blocklevel >= 2 might be provided
   *       in a future version.
   *
   * \tparam BCRSMatrix  Type of a DUNE-ISTL BCRSMatrix whose eigenvalues
   *                     shall be considered; is assumed to have blocklevel
   *                     2 with square blocks.
   * \tparam BlockVector Type of the associated vectors; compatible with
   *                     the rows of a BCRSMatrix object and its columns.
   *
   * \author Sebastian Westerheide.
   */
  template <typename BCRSMatrix, typename BlockVector>
  class PowerIteration_Algorithms
  {
  protected:
    // Type definitions for type of iteration operator (m_ - mu_*I)
    typedef typename Dune::MatrixAdapter<BCRSMatrix,BlockVector,BlockVector>
      MatrixOperator;
    typedef Impl::ScalingLinearOperator<BlockVector> ScalingOperator;
    typedef Impl::LinearOperatorSum<MatrixOperator,ScalingOperator> OperatorSum;

  public:
    //! Type of underlying field
    typedef typename BlockVector::field_type Real;

    //! Type of iteration operator (m_ - mu_*I)
    typedef OperatorSum IterationOperator;

  public:
    /**
     * \brief Construct from required parameters.
     *
     * \param[in] m               The square DUNE-ISTL BCRSMatrix whose
     *                            eigenvalues shall be considered.
     * \param[in] nIterationsMax  The maximum number of iterations allowed.
     * \param[in] verbosity_level Verbosity setting;
     *                            >= 1: algorithms print a preamble and
     *                                  the final result,
     *                            >= 2: algorithms print information on
     *                                  each iteration,
     *                            >= 3: the final result output includes
     *                                  the approximated eigenvector.
     */
    PowerIteration_Algorithms (const BCRSMatrix& m,
                               const unsigned int nIterationsMax = 1000,
                               const unsigned int verbosity_level = 0)
      : m_(m), nIterationsMax_(nIterationsMax),
        verbosity_level_(verbosity_level),
        mu_(0.0),
        matrixOperator_(m_),
        scalingOperator_(-1.0,mu_),
        itOperator_(matrixOperator_,scalingOperator_),
        nIterations_(0),
        title_("    PowerIteration_Algorithms: "),
        blank_(title_.length(),' ')
    {
      // assert that BCRSMatrix type has blocklevel 2
      static_assert
        (blockLevel<BCRSMatrix>() == 2,
         "Only BCRSMatrices with blocklevel 2 are supported.");

      // assert that BCRSMatrix type has square blocks
      static_assert
        (BCRSMatrix::block_type::rows == BCRSMatrix::block_type::cols,
         "Only BCRSMatrices with square blocks are supported.");

      // assert that m_ is square
      const int nrows = m_.M() * BCRSMatrix::block_type::rows;
      const int ncols = m_.N() * BCRSMatrix::block_type::cols;
      if (nrows != ncols)
        DUNE_THROW(Dune::ISTLError,"Matrix is not square ("
                   << nrows << "x" << ncols << ").");
    }

    //! disallow copying (default copy constructor does a shallow copy,
    //! if copying was required a deep copy would have to be implemented
    //! due to member variables which hold a dynamically allocated object)
    PowerIteration_Algorithms (const PowerIteration_Algorithms&) = delete;

    //! disallow copying (default assignment operator does a shallow copy,
    //! if copying was required a deep copy would have to be implemented
    //! due to member variables which hold a dynamically allocated object)
    PowerIteration_Algorithms&
      operator= (const PowerIteration_Algorithms&) = delete;

    /**
     * \brief Perform the power iteration algorithm to compute an approximation
     *        lambda of the dominant (i.e. largest magnitude) eigenvalue and
     *        the corresponding approximation x of an associated eigenvector.
     *
     * \param[in]     epsilon The target residual norm.
     * \param[out]    lambda  The approximated dominant eigenvalue.
     * \param[in,out] x       The associated approximated eigenvector;
     *                        shall be initialized with an estimate
     *                        for an eigenvector associated with the
     *                        eigenvalue which shall be approximated.
     */
    inline void applyPowerIteration (const Real& epsilon,
                                     BlockVector& x, Real& lambda) const
    {
      // print verbosity information
      if (verbosity_level_ > 0)
        std::cout << title_
                  << "Performing power iteration approximating "
                  << "the dominant eigenvalue." << std::endl;

      // allocate memory for auxiliary variables
      BlockVector y(x);
      BlockVector temp(x);

      // perform power iteration
      x *= (1.0 / x.two_norm());
      m_.mv(x,y);
      Real r_norm = std::numeric_limits<Real>::max();
      nIterations_ = 0;
      while (r_norm > epsilon)
      {
        // update and check number of iterations
        if (++nIterations_ > nIterationsMax_)
          DUNE_THROW(Dune::ISTLError,"Power iteration did not converge "
                     << "in " << nIterationsMax_ << " iterations "
                     << "(║residual║_2 = " << r_norm << ", epsilon = "
                     << epsilon << ").");

        // do one iteration of the power iteration algorithm
        // (use that y = m_ * x)
        x = y;
        x *= (1.0 / y.two_norm());

        // get approximated eigenvalue lambda via the Rayleigh quotient
        m_.mv(x,y);
        lambda = x * y;

        // get norm of residual (use that y = m_ * x)
        temp = y;
        temp.axpy(-lambda,x);
        r_norm = temp.two_norm();

        // print verbosity information
        if (verbosity_level_ > 1)
          std::cout << blank_ << std::left
                    << "iteration " << std::setw(3) << nIterations_
                    << " (║residual║_2 = " << std::setw(11) << r_norm
                    << "): λ = " << lambda << std::endl
                    << std::resetiosflags(std::ios::left);
      }

      // print verbosity information
      if (verbosity_level_ > 0)
      {
        std::cout << blank_ << "Result ("
                  << "#iterations = " << nIterations_ << ", "
                  << "║residual║_2 = " << r_norm << "): "
                  << "λ = " << lambda << std::endl;
        if (verbosity_level_ > 2)
        {
          // print approximated eigenvector via DUNE-ISTL I/O methods
          Dune::printvector(std::cout,x,blank_+"x",blank_+"row");
        }
      }
    }

    /**
     * \brief Perform the inverse iteration algorithm to compute an approximation
     *        lambda of the least dominant (i.e. smallest magnitude) eigenvalue
     *        and the corresponding approximation x of an associated eigenvector.
     *
     * \tparam ISTLLinearSolver    Type of a DUNE-ISTL InverseOperator
     *                             which shall be used as a linear solver.
     * \tparam avoidLinSolverCrime The less accurate the linear solver is,
     *                             the more corrupted gets the implemented
     *                             computation of lambda and its associated
     *                             residual. Setting this mode can help
     *                             increasing their accuracy at the cost of
     *                             a bit of efficiency which is beneficial
     *                             e.g. when using a very inexact linear
     *                             solver. Defaults to false.
     *
     * \param[in]     epsilon The target residual norm.
     * \param[in]     solver  The DUNE-ISTL InverseOperator which shall
     *                        be used as a linear solver; is assumed to
     *                        be constructed using the linear operator
     *                        returned by getIterationOperator() (resp.
     *                        matrix returned by getIterationMatrix()).
     * \param[out]    lambda  The approximated least dominant eigenvalue.
     * \param[in,out] x       The associated approximated eigenvector;
     *                        shall be initialized with an estimate
     *                        for an eigenvector associated with the
     *                        eigenvalue which shall be approximated.
     */
    template <typename ISTLLinearSolver,
              bool avoidLinSolverCrime = false>
    inline void applyInverseIteration (const Real& epsilon,
                                       ISTLLinearSolver& solver,
                                       BlockVector& x, Real& lambda) const
    {
      constexpr Real gamma = 0.0;
      applyInverseIteration(gamma,epsilon,solver,x,lambda);
    }

    /**
     * \brief Perform the inverse iteration with shift algorithm to compute an
     *        approximation lambda of the eigenvalue closest to a given shift
     *        and the corresponding approximation x of an associated eigenvector.
     *
     * \tparam ISTLLinearSolver    Type of a DUNE-ISTL InverseOperator
     *                             which shall be used as a linear solver.
     * \tparam avoidLinSolverCrime The less accurate the linear solver is,
     *                             the more corrupted gets the implemented
     *                             computation of lambda and its associated
     *                             residual. Setting this mode can help
     *                             increasing their accuracy at the cost of
     *                             a bit of efficiency which is beneficial
     *                             e.g. when using a very inexact linear
     *                             solver. Defaults to false.
     *
     * \param[in]     gamma   The shift.
     * \param[in]     epsilon The target residual norm.
     * \param[in]     solver  The DUNE-ISTL InverseOperator which shall
     *                        be used as a linear solver; is assumed to
     *                        be constructed using the linear operator
     *                        returned by getIterationOperator() (resp.
     *                        matrix returned by getIterationMatrix()).
     * \param[out]    lambda  The approximated eigenvalue closest to gamma.
     * \param[in,out] x       The associated approximated eigenvector;
     *                        shall be initialized with an estimate
     *                        for an eigenvector associated with the
     *                        eigenvalue which shall be approximated.
     */
    template <typename ISTLLinearSolver,
              bool avoidLinSolverCrime = false>
    inline void applyInverseIteration (const Real& gamma,
                                       const Real& epsilon,
                                       ISTLLinearSolver& solver,
                                       BlockVector& x, Real& lambda) const
    {
      // print verbosity information
      if (verbosity_level_ > 0)
      {
        std::cout << title_;
        if (gamma == 0.0)
          std::cout << "Performing inverse iteration approximating "
                    << "the least dominant eigenvalue." << std::endl;
        else
          std::cout << "Performing inverse iteration with shift "
                    << "gamma = " << gamma << " approximating the "
                    << "eigenvalue closest to gamma." << std::endl;
      }

      // initialize iteration operator,
      // initialize iteration matrix when needed
      updateShiftMu(gamma,solver);

      // allocate memory for linear solver statistics
      Dune::InverseOperatorResult solver_statistics;

      // allocate memory for auxiliary variables
      BlockVector y(x);
      Real y_norm;
      BlockVector temp(x);

      // perform inverse iteration with shift
      x *= (1.0 / x.two_norm());
      Real r_norm = std::numeric_limits<Real>::max();
      nIterations_ = 0;
      while (r_norm > epsilon)
      {
        // update and check number of iterations
        if (++nIterations_ > nIterationsMax_)
          DUNE_THROW(Dune::ISTLError,"Inverse iteration "
                     << (gamma != 0.0 ? "with shift " : "") << "did not "
                     << "converge in " << nIterationsMax_ << " iterations "
                     << "(║residual║_2 = " << r_norm << ", epsilon = "
                     << epsilon << ").");

        // do one iteration of the inverse iteration with shift algorithm,
        // part 1: solve (m_ - gamma*I) * y = x for y
        // (protect x from being changed)
        temp = x;
        solver.apply(y,temp,solver_statistics);

        // get norm of y
        y_norm = y.two_norm();

        // compile time switch between accuracy and efficiency
        if (avoidLinSolverCrime)
        {
          // get approximated eigenvalue lambda via the Rayleigh quotient
          // (use that x_new = y / y_norm)
          m_.mv(y,temp);
          lambda = (y * temp) / (y_norm * y_norm);

          // get norm of residual
          // (use that x_new = y / y_norm, additionally use that temp = m_ * y)
          temp.axpy(-lambda,y);
          r_norm = temp.two_norm() / y_norm;
        }
        else
        {
          // get approximated eigenvalue lambda via the Rayleigh quotient
          // (use that x_new = y / y_norm and use that (m_ - gamma*I) * y = x)
          lambda = gamma + (y * x) / (y_norm * y_norm);

          // get norm of residual
          // (use that x_new = y / y_norm and use that (m_ - gamma*I) * y = x)
          temp = x; temp.axpy(gamma-lambda,y);
          r_norm = temp.two_norm() / y_norm;
        }

        // do one iteration of the inverse iteration with shift algorithm,
        // part 2: update x
        x = y;
        x *= (1.0 / y_norm);

        // print verbosity information
        if (verbosity_level_ > 1)
          std::cout << blank_ << std::left
                    << "iteration " << std::setw(3) << nIterations_
                    << " (║residual║_2 = " << std::setw(11) << r_norm
                    << "): λ = " << lambda << std::endl
                    << std::resetiosflags(std::ios::left);
      }

      // print verbosity information
      if (verbosity_level_ > 0)
      {
        std::cout << blank_ << "Result ("
                  << "#iterations = " << nIterations_ << ", "
                  << "║residual║_2 = " << r_norm << "): "
                  << "λ = " << lambda << std::endl;
        if (verbosity_level_ > 2)
        {
          // print approximated eigenvector via DUNE-ISTL I/O methods
          Dune::printvector(std::cout,x,blank_+"x",blank_+"row");
        }
      }
    }

    /**
     * \brief Perform the Rayleigh quotient iteration algorithm to compute
     *        an approximation lambda of an eigenvalue and the corresponding
     *        approximation x of an associated eigenvector.
     *
     * \tparam ISTLLinearSolver    Type of a DUNE-ISTL InverseOperator
     *                             which shall be used as a linear solver.
     * \tparam avoidLinSolverCrime The less accurate the linear solver is,
     *                             the more corrupted gets the implemented
     *                             computation of lambda and its associated
     *                             residual. Setting this mode can help
     *                             increasing their accuracy at the cost of
     *                             a bit of efficiency which is beneficial
     *                             e.g. when using a very inexact linear
     *                             solver. Defaults to false.
     *
     * \param[in]     epsilon The target residual norm.
     * \param[in]     solver  The DUNE-ISTL InverseOperator which shall
     *                        be used as a linear solver; is assumed to
     *                        be constructed using the linear operator
     *                        returned by getIterationOperator() (resp.
     *                        matrix returned by getIterationMatrix()).
     * \param[in,out] lambda  The approximated eigenvalue;
     *                        shall be initialized with an estimate for
     *                        the eigenvalue which shall be approximated.
     * \param[in,out] x       The associated approximated eigenvector;
     *                        shall be initialized with an estimate
     *                        for an eigenvector associated with the
     *                        eigenvalue which shall be approximated.
     */
    template <typename ISTLLinearSolver,
              bool avoidLinSolverCrime = false>
    inline void applyRayleighQuotientIteration (const Real& epsilon,
                                                ISTLLinearSolver& solver,
                                                BlockVector& x, Real& lambda) const
    {
      // print verbosity information
      if (verbosity_level_ > 0)
        std::cout << title_
                  << "Performing Rayleigh quotient iteration for "
                  << "estimated eigenvalue " << lambda << "." << std::endl;

      // allocate memory for linear solver statistics
      Dune::InverseOperatorResult solver_statistics;

      // allocate memory for auxiliary variables
      BlockVector y(x);
      Real y_norm;
      Real lambda_update;
      BlockVector temp(x);

      // perform Rayleigh quotient iteration
      x *= (1.0 / x.two_norm());
      Real r_norm = std::numeric_limits<Real>::max();
      nIterations_ = 0;
      while (r_norm > epsilon)
      {
        // update and check number of iterations
        if (++nIterations_ > nIterationsMax_)
          DUNE_THROW(Dune::ISTLError,"Rayleigh quotient iteration did not "
                     << "converge in " << nIterationsMax_ << " iterations "
                     << "(║residual║_2 = " << r_norm << ", epsilon = "
                     << epsilon << ").");

        // update iteration operator,
        // update iteration matrix when needed
        updateShiftMu(lambda,solver);

        // do one iteration of the Rayleigh quotient iteration algorithm,
        // part 1: solve (m_ - lambda*I) * y = x for y
        // (protect x from being changed)
        temp = x;
        solver.apply(y,temp,solver_statistics);

        // get norm of y
        y_norm = y.two_norm();

        // compile time switch between accuracy and efficiency
        if (avoidLinSolverCrime)
        {
          // get approximated eigenvalue lambda via the Rayleigh quotient
          // (use that x_new = y / y_norm)
          m_.mv(y,temp);
          lambda = (y * temp) / (y_norm * y_norm);

          // get norm of residual
          // (use that x_new = y / y_norm, additionally use that temp = m_ * y)
          temp.axpy(-lambda,y);
          r_norm = temp.two_norm() / y_norm;
        }
        else
        {
          // get approximated eigenvalue lambda via the Rayleigh quotient
          // (use that x_new = y / y_norm and use that (m_ - lambda_old*I) * y = x)
          lambda_update = (y * x) / (y_norm * y_norm);
          lambda += lambda_update;

          // get norm of residual
          // (use that x_new = y / y_norm and use that (m_ - lambda_old*I) * y = x)
          temp = x; temp.axpy(-lambda_update,y);
          r_norm = temp.two_norm() / y_norm;
        }

        // do one iteration of the Rayleigh quotient iteration algorithm,
        // part 2: update x
        x = y;
        x *= (1.0 / y_norm);

        // print verbosity information
        if (verbosity_level_ > 1)
          std::cout << blank_ << std::left
                    << "iteration " << std::setw(3) << nIterations_
                    << " (║residual║_2 = " << std::setw(11) << r_norm
                    << "): λ = " << lambda << std::endl
                    << std::resetiosflags(std::ios::left);
      }

      // print verbosity information
      if (verbosity_level_ > 0)
      {
        std::cout << blank_ << "Result ("
                  << "#iterations = " << nIterations_ << ", "
                  << "║residual║_2 = " << r_norm << "): "
                  << "λ = " << lambda << std::endl;
        if (verbosity_level_ > 2)
        {
          // print approximated eigenvector via DUNE-ISTL I/O methods
          Dune::printvector(std::cout,x,blank_+"x",blank_+"row");
        }
      }
    }

    /**
     * \brief Perform the "two-level iterative method for eigenvalue calculations
     *        (TLIME)" iteration algorithm presented in [Szyld, 1988] to compute
     *        an approximation lambda of an eigenvalue and the corresponding
     *        approximation x of an associated eigenvector.
     *
     * The algorithm combines the inverse iteration with shift and the Rayleigh
     * quotient iteration in order to compute an eigenvalue in a given interval
     * J = (gamma - eta, gamma + eta). It guarantees that if an eigenvalue exists
     * in J, the method will converge to an eigenvalue in J, while exploiting
     * the cubic convergence of the Rayleigh quotient iteration, but without its
     * drawback that - depending on the initial vector - it can converge to an
     * arbitrary eigenvalue of the matrix. When J is free of eigenvalues, the
     * method will determine this fact and converge linearly to the eigenvalue
     * closest to J.
     *
     * \tparam ISTLLinearSolver    Type of a DUNE-ISTL InverseOperator
     *                             which shall be used as a linear solver.
     * \tparam avoidLinSolverCrime The less accurate the linear solver is,
     *                             the more corrupted gets the implemented
     *                             computation of lambda and its associated
     *                             residual. Setting this mode can help
     *                             increasing their accuracy at the cost of
     *                             a bit of efficiency which is beneficial
     *                             e.g. when using a very inexact linear
     *                             solver. Defaults to false.
     *
     * \param[in]     gamma   An estimate for the eigenvalue which shall
     *                        be approximated.
     * \param[in]     eta     Radius around gamma in which the eigenvalue
     *                        is expected.
     * \param[in]     epsilon The target norm of the residual with respect
     *                        to the Rayleigh quotient.
     * \param[in]     solver  The DUNE-ISTL InverseOperator which shall
     *                        be used as a linear solver; is assumed to
     *                        be constructed using the linear operator
     *                        returned by getIterationOperator() (resp.
     *                        matrix returned by getIterationMatrix()).
     * \param[in]     delta   The target relative change of the Rayleigh
     *                        quotient, indicating that inverse iteration
     *                        has become stationary and switching to Rayleigh
     *                        quotient iteration is appropriate; is only
     *                        considered if J is free of eigenvalues.
     * \param[in]     m       The minimum number of inverse iterations before
     *                        switching to Rayleigh quotient iteration; is
     *                        only considered if J is free of eigenvalues.
     * \param[out]    extrnl  If true, the interval J is free of eigenvalues;
     *                        the approximated eigenvalue-eigenvector pair
     *                        (lambda,x_s) then corresponds to the eigenvalue
     *                        closest to J.
     * \param[out]    lambda  The approximated eigenvalue.
     * \param[in,out] x       The associated approximated eigenvector;
     *                        shall be initialized with an estimate
     *                        for an eigenvector associated with the
     *                        eigenvalue which shall be approximated.
     */
    template <typename ISTLLinearSolver,
              bool avoidLinSolverCrime = false>
    inline void applyTLIMEIteration (const Real& gamma, const Real& eta,
                                     const Real& epsilon,
                                     ISTLLinearSolver& solver,
                                     const Real& delta, const std::size_t& m,
                                     bool& extrnl,
                                     BlockVector& x, Real& lambda) const
    {
      // use same variable names as in [Szyld, 1988]
      BlockVector& x_s = x;
      Real& mu_s = lambda;

      // print verbosity information
      if (verbosity_level_ > 0)
        std::cout << title_
                  << "Performing TLIME iteration for "
                  << "estimated eigenvalue in the "
                  << "interval (" << gamma - eta << ","
                  << gamma + eta << ")." << std::endl;

      // allocate memory for linear solver statistics
      Dune::InverseOperatorResult solver_statistics;

      // allocate memory for auxiliary variables
      bool doRQI;
      Real mu;
      BlockVector y(x_s);
      Real omega;
      Real mu_s_old;
      Real mu_s_update;
      BlockVector temp(x_s);
      Real q_norm, r_norm;

      // perform TLIME iteration
      x_s *= (1.0 / x_s.two_norm());
      extrnl = true;
      doRQI = false;
      r_norm = std::numeric_limits<Real>::max();
      nIterations_ = 0;
      while (r_norm > epsilon)
      {
        // update and check number of iterations
        if (++nIterations_ > nIterationsMax_)
          DUNE_THROW(Dune::ISTLError,"TLIME iteration did not "
                     << "converge in " << nIterationsMax_
                     << " iterations (║residual║_2 = " << r_norm
                     << ", epsilon = " << epsilon << ").");

        // set shift for next iteration according to inverse iteration
        // with shift (II) resp. Rayleigh quotient iteration (RQI)
        if (doRQI)
          mu = mu_s;
        else
          mu = gamma;

        // update II/RQI iteration operator,
        // update II/RQI iteration matrix when needed
        updateShiftMu(mu,solver);

        // do one iteration of the II/RQI algorithm,
        // part 1: solve (m_ - mu*I) * y = x for y
        temp = x_s;
        solver.apply(y,temp,solver_statistics);

        // do one iteration of the II/RQI algorithm,
        // part 2: compute omega
        omega = (1.0 / y.two_norm());

        // backup the old Rayleigh quotient
        mu_s_old = mu_s;

        // compile time switch between accuracy and efficiency
        if (avoidLinSolverCrime)
        {
          // update the Rayleigh quotient mu_s, i.e. the approximated eigenvalue
          // (use that x_new = y * omega)
          m_.mv(y,temp);
          mu_s = (y * temp) * (omega * omega);

          // get norm of "the residual with respect to the shift used by II",
          // use normal representation of q
          // (use that x_new = y * omega, use that temp = m_ * y)
          temp.axpy(-gamma,y);
          q_norm = temp.two_norm() * omega;

          // get norm of "the residual with respect to the Rayleigh quotient"
          r_norm = q_norm*q_norm - (gamma-mu_s)*(gamma-mu_s);
          // prevent that truncation errors invalidate the norm
          // (we don't want to calculate sqrt of a negative number)
          if (r_norm >= 0)
          {
            // use relation between the norms of r and q for efficiency
            r_norm = std::sqrt(r_norm);
          }
          else
          {
            // use relation between r and q
            // (use that x_new = y * omega, use that temp = (m_ - gamma*I) * y = q / omega)
            temp.axpy(gamma-mu_s,y);
            r_norm = temp.two_norm() * omega;
          }
        }
        else
        {
          // update the Rayleigh quotient mu_s, i.e. the approximated eigenvalue
          if (!doRQI)
          {
            // (use that x_new = y * omega, additionally use that (m_ - gamma*I) * y = x_s)
            mu_s = gamma + (y * x_s) * (omega * omega);
          }
          else
          {
            // (use that x_new = y * omega, additionally use that (m_ - mu_s_old*I) * y = x_s)
            mu_s_update = (y * x_s) * (omega * omega);
            mu_s += mu_s_update;
          }

          // get norm of "the residual with respect to the shift used by II"
          if (!doRQI)
          {
            // use special representation of q in the II case
            // (use that x_new = y * omega, additionally use that (m_ - gamma*I) * y = x_s)
            q_norm = omega;
          }
          else
          {
            // use special representation of q in the RQI case
            // (use that x_new = y * omega, additionally use that (m_ - mu_s_old*I) * y = x_s)
            temp = x_s; temp.axpy(mu_s-gamma,y);
            q_norm = temp.two_norm() * omega;
          }

          // get norm of "the residual with respect to the Rayleigh quotient"
          // don't use efficient relation between the norms of r and q, as
          // this relation seems to yield a less accurate r_norm in the case
          // where linear solver crime is admitted
          if (!doRQI)
          {
            // (use that x_new = y * omega and use that (m_ - gamma*I) * y = x_s)
            temp = x_s; temp.axpy(gamma-lambda,y);
            r_norm = temp.two_norm() * omega;
          }
          else
          {
            // (use that x_new = y * omega and use that (m_ - mu_s_old*I) * y = x_s)
            temp = x_s; temp.axpy(-mu_s_update,y);
            r_norm = temp.two_norm() * omega;
          }
        }

        // do one iteration of the II/RQI algorithm,
        // part 3: update x
        x_s = y; x_s *= omega;

        // // for relative residual norm mode, scale with mu_s^{-1}
        // r_norm /= std::abs(mu_s);

        // print verbosity information
        if (verbosity_level_ > 1)
          std::cout << blank_ << "iteration "
                    << std::left << std::setw(3) << nIterations_
                    << " (" << (doRQI ? "RQI," : "II, ")
                    << " " << (doRQI ? "—>" : "  ") << " "
                    << "║r║_2 = " << std::setw(11) << r_norm
                    << ", " << (doRQI ? "  " : "—>") << " "
                    << "║q║_2 = " << std::setw(11) << q_norm
                    << "): λ = " << lambda << std::endl
                    << std::resetiosflags(std::ios::left);

        // check if the eigenvalue closest to gamma lies in J
        if (!doRQI && q_norm < eta)
        {
          // J is not free of eigenvalues
          extrnl = false;

          // by theory we know now that mu_s also lies in J
          assert(std::abs(mu_s-gamma) < eta);

          // switch to RQI
          doRQI = true;
        }

        // revert to II if J is not free of eigenvalues but
        // at some point mu_s falls back again outside J
        if (!extrnl && doRQI && std::abs(mu_s-gamma) >= eta)
          doRQI = false;

        // if eigenvalue closest to gamma does not lie in J use RQI
        // solely to accelerate the convergence to this eigenvalue
        // when II has become stationary
        if (extrnl && !doRQI)
        {
          // switch to RQI if the relative change of the Rayleigh
          // quotient indicates that II has become stationary
          if (nIterations_ >= m &&
              std::abs(mu_s - mu_s_old) / std::abs(mu_s) < delta)
            doRQI = true;
        }
      }

      // // compute final residual and lambda again (paranoia....)
      // m_.mv(x_s,temp);
      // mu_s = x_s * temp;
      // temp.axpy(-mu_s,x_s);
      // r_norm = temp.two_norm();
      // // r_norm /= std::abs(mu_s);

      // print verbosity information
      if (verbosity_level_ > 0)
      {
        if (extrnl)
          std::cout << blank_ << "Interval "
                    << "(" << gamma - eta << "," << gamma + eta
                    << ") is free of eigenvalues, approximating "
                    << "the closest eigenvalue." << std::endl;
        std::cout << blank_ << "Result ("
                  << "#iterations = " << nIterations_ << ", "
                  << "║residual║_2 = " << r_norm << "): "
                  << "λ = " << lambda << std::endl;
        if (verbosity_level_ > 2)
        {
          // print approximated eigenvector via DUNE-ISTL I/O methods
          Dune::printvector(std::cout,x,blank_+"x",blank_+"row");
        }
      }
    }

    /**
     * \brief Return the iteration operator (m_ - mu_*I).
     *
     * The linear operator returned by this method shall be used
     * to create the linear solver object. For linear solvers or
     * preconditioners which require that the matrix is provided
     * explicitly use getIterationMatrix() instead/additionally.
     */
    inline IterationOperator& getIterationOperator ()
    {
      // return iteration operator
      return itOperator_;
    }

    /**
     * \brief Return the iteration matrix (m_ - mu_*I), provided
     *        on demand when needed (e.g. for direct solvers or
     *        preconditioning).
     *
     * The matrix returned by this method shall be used to create
     * the linear solver object if it requires that the matrix is
     * provided explicitly. For linear solvers which operate
     * completely matrix free use getIterationOperator() instead.
     *
     * \note Calling this method creates a new DUNE-ISTL
     *       BCRSMatrix object which requires as much memory as
     *       the matrix whose eigenvalues shall be considered.
     */
    inline const BCRSMatrix& getIterationMatrix () const
    {
      // create iteration matrix on demand
      if (!itMatrix_)
        itMatrix_ = std::make_unique<BCRSMatrix>(m_);

      // return iteration matrix
      return *itMatrix_;
    }

    /**
     * \brief Return the number of iterations in last application
     *        of an algorithm.
     */
    inline unsigned int getIterationCount () const
    {
      if (nIterations_ == 0)
        DUNE_THROW(Dune::ISTLError,"No algorithm applied, yet.");

      return nIterations_;
    }

  protected:
    /**
     * \brief Update shift mu_, i.e. update iteration operator/matrix
     *        (m_ - mu_*I).
     *
     * \note Does nothing if new shift equals the old one.
     *
     * \tparam ISTLLinearSolver Type of a DUNE-ISTL InverseOperator
     *                          which is used as a linear solver.
     *
     * \param[in] mu     The new shift.
     * \param[in] solver The DUNE-ISTL InverseOperator which is used
     *                   as a linear solver.
     *
     */
    template <typename ISTLLinearSolver>
    inline void updateShiftMu (const Real& mu,
                               ISTLLinearSolver& solver) const
    {
      // do nothing if new shift equals the old one
      if (mu == mu_) return;

      // update shift mu_, i.e. update iteration operator
      mu_ = mu;

      // update iteration matrix when needed
      if (itMatrix_)
      {
        // iterate over entries in iteration matrix diagonal
        constexpr int rowBlockSize = BCRSMatrix::block_type::rows;
        constexpr int colBlockSize = BCRSMatrix::block_type::cols;
        for (typename BCRSMatrix::size_type i = 0;
             i < itMatrix_->M()*rowBlockSize; ++i)
        {
          // access m_[i,i] where i is the flat index of a row/column
          const Real& m_entry = m_
            [i/rowBlockSize][i/colBlockSize][i%rowBlockSize][i%colBlockSize];
          // access *itMatrix[i,i] where i is the flat index of a row/column
          Real& entry = (*itMatrix_)
            [i/rowBlockSize][i/colBlockSize][i%rowBlockSize][i%colBlockSize];
          // change current entry in iteration matrix diagonal
          entry = m_entry - mu_;
        }
        // notify linear solver about change of the iteration matrix object
        SolverHelper<ISTLLinearSolver,BCRSMatrix>::setMatrix
          (solver,*itMatrix_);
      }
    }

  protected:
    // parameters related to iterative eigenvalue algorithms
    const BCRSMatrix& m_;
    const unsigned int nIterationsMax_;

    // verbosity setting
    const unsigned int verbosity_level_;

    // shift mu_ used by iteration operator/matrix (m_ - mu_*I)
    mutable Real mu_;

    // iteration operator (m_ - mu_*I), passing shift mu_ by reference
    const MatrixOperator matrixOperator_;
    const ScalingOperator scalingOperator_;
    OperatorSum itOperator_;

    // iteration matrix (m_ - mu_*I), provided on demand when needed
    // (e.g. for preconditioning)
    mutable std::unique_ptr<BCRSMatrix> itMatrix_;

    // memory for storing temporary variables (mutable as they shall
    // just be effectless auxiliary variables of the const apply*(...)
    // methods)
    mutable unsigned int nIterations_;

    // constants for printing verbosity information
    const std::string title_;
    const std::string blank_;
  };

  /** @} */

}  // namespace Dune

#endif  // DUNE_ISTL_EIGENVALUE_POWERITERATION_HH
