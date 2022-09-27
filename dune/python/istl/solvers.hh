// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_PYTHON_ISTL_SOLVER_HH
#define DUNE_PYTHON_ISTL_SOLVER_HH

#include <dune/common/typeutilities.hh>
#include <dune/common/version.hh>

#include <dune/istl/solver.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>

#include <dune/python/istl/preconditioners.hh>

#include <dune/python/pybind11/pybind11.h>

namespace Dune
{

  namespace Python
  {

    // registerInverseOperator
    // -----------------------

    template< class Solver, class... options >
    inline void registerInverseOperator ( pybind11::class_< Solver, options... > cls )
    {
      typedef typename Solver::domain_type Domain;
      typedef typename Solver::range_type Range;

      using pybind11::operator""_a;

      cls.def( "__call__", [] ( Solver &self, Domain &x, Range &b, double reduction ) {
          InverseOperatorResult result;
          self.apply( x, b, reduction, result );
          return std::make_tuple( result.iterations, result.reduction, result.converged, result.conv_rate, result.elapsed );
        }, "x"_a, "b"_a, "reduction"_a,
        R"doc(
          Solve linear system

          Args:
              x:          solution of linear system
              b:          right hand side of the system
              reduction:  factor to reduce the defect by

          Returns: (iterations, reduction, converged, conv_rate, elapsed)
              iterations:  number of iterations performed
              reduction:   actual factor, the error has been reduced by
              converged:   True, if the solver has achieved its reduction requirements
              conv_rate:   rate of convergence
              elapsed:     time in seconds used to solve the linear system

          Note:
              - If the reduction is omitted, the default value of the solver is used.
              - For iterative solvers, the solution must be initialized to the starting point.
              - The right hand side b will be replaced by the residual.
        )doc" );

      cls.def( "__call__", [] ( Solver &self, Domain &x, Range &b ) {
          InverseOperatorResult result;
          self.apply( x, b, result );
          return std::make_tuple( result.iterations, result.reduction, result.converged, result.conv_rate, result.elapsed );
        }, "x"_a, "b"_a );

      cls.def_property_readonly( "category", [] ( const Solver &self ) { return self.category(); },
        R"doc(
          Obtain category of the linear solver
        )doc" );

      cls.def( "asPreconditioner", [] ( Solver &self ) {
          return new InverseOperator2Preconditioner< Solver >( self );
        }, pybind11::keep_alive< 0, 1 >(),
        R"doc(
          Convert linear solver into preconditioner
        )doc" );
    }



    namespace detail
    {

      // registerEndomorphismSolvers
      // ---------------------------

      template< class X, class Y, class... options >
      inline std::enable_if_t< std::is_same< X, Y >::value >
      registerEndomorphismSolvers ( pybind11::module module, pybind11::class_< LinearOperator< X, Y >, options... >, PriorityTag< 1 > )
      {
        typedef Dune::InverseOperator< X, Y > Solver;

        using pybind11::operator""_a;

        pybind11::options opts;
        opts.disable_function_signatures();

        module.def( "LoopSolver", [] ( LinearOperator< X, X > &op, Preconditioner< X, X > &prec, double reduction, int maxit, int verbose ) {
            return static_cast< Solver * >( new Dune::LoopSolver< X >( op, prec, reduction, maxit, verbose ) );
          }, "operator"_a, "preconditioner"_a, "reduction"_a, "maxIterations"_a = std::numeric_limits< int >::max(), "verbose"_a = 0, pybind11::keep_alive< 0, 1 >(), pybind11::keep_alive< 0, 2 >(),
          R"doc(
            Loop solver

            Args:
                operator:        operator to invert
                preconditioner:  preconditioner to use (i.e., apprixmate inverse of the operator)
                reduction:       factor to reduce the defect by
                maxIterations:   maximum number of iterations to perform
                verbose:         verbosity level (0 = quiet, 1 = summary, 2 = verbose)

            Returns:
                ISTL Loop solver

            Note:
                The loop solver will apply the preconditioner once in each step.
          )doc" );

        module.def( "GradientSolver", [] ( LinearOperator< X, X > &op, Preconditioner< X, X > &prec, double reduction, int maxit, int verbose ) {
            return static_cast< Solver * >( new Dune::GradientSolver< X >( op, prec, reduction, maxit, verbose ) );
          }, "operator"_a, "preconditioner"_a, "reduction"_a, "maxIterations"_a = std::numeric_limits< int >::max(), "verbose"_a = 0, pybind11::keep_alive< 0, 1 >(), pybind11::keep_alive< 0, 2 >(),
          R"doc(
            Gradient iterative solver

            Args:
                operator:        operator to invert
                preconditioner:  preconditioner to use (i.e., apprixmate inverse of the operator)
                reduction:       factor to reduce the defect by
                maxIterations:   maximum number of iterations to perform
                verbose:         verbosity level (0 = quiet, 1 = summary, 2 = verbose)

            Returns:
                ISTL Gradient solver

            Note:
                This method is also know as steepest descend method.
          )doc" );

        module.def( "CGSolver", [] ( LinearOperator< X, X > &op, Preconditioner< X, X > &prec, double reduction, int maxit, int verbose ) {
            return static_cast< Solver * >( new Dune::CGSolver< X >( op, prec, reduction, maxit, verbose ) );
          }, "operator"_a, "preconditioner"_a, "reduction"_a, "maxIterations"_a = std::numeric_limits< int >::max(), "verbose"_a = 0, pybind11::keep_alive< 0, 1 >(), pybind11::keep_alive< 0, 2 >(),
          R"doc(
            Conjugate gradient iterative solver

            Args:
                operator:        operator to invert
                preconditioner:  preconditioner to use (i.e., apprixmate inverse of the operator)
                reduction:       factor to reduce the defect by
                maxIterations:   maximum number of iterations to perform
                verbose:         verbosity level (0 = quiet, 1 = summary, 2 = verbose)

            Returns:
                ISTL Conjugate gradient solver

            Note:
                The conjucate gradient method can only be applied if the operator and the preconditioner are both symmetric and positive definite.
          )doc" );

        module.def( "BiCGSTABSolver", [] ( LinearOperator< X, X > &op, Preconditioner< X, X > &prec, double reduction, int maxit, int verbose ) {
            return static_cast< Solver * >( new Dune::CGSolver< X >( op, prec, reduction, maxit, verbose ) );
          }, "operator"_a, "preconditioner"_a, "reduction"_a, "maxIterations"_a = std::numeric_limits< int >::max(), "verbose"_a = 0, pybind11::keep_alive< 0, 1 >(), pybind11::keep_alive< 0, 2 >(),
          R"doc(
            Biconjugate gradient stabilized iterative solver

            Args:
                operator:        operator to invert
                preconditioner:  preconditioner to use (i.e., apprixmate inverse of the operator)
                reduction:       factor to reduce the defect by
                maxIterations:   maximum number of iterations to perform
                verbose:         verbosity level (0 = quiet, 1 = summary, 2 = verbose)

            Returns:
                ISTL Biconjugate gradient stabilized solver
          )doc" );

        module.def( "MinResSolver", [] ( LinearOperator< X, X > &op, Preconditioner< X, X > &prec, double reduction, int maxit, int verbose ) {
            return static_cast< Solver * >( new Dune::CGSolver< X >( op, prec, reduction, maxit, verbose ) );
          }, "operator"_a, "preconditioner"_a, "reduction"_a, "maxIterations"_a = std::numeric_limits< int >::max(), "verbose"_a = 0, pybind11::keep_alive< 0, 1 >(), pybind11::keep_alive< 0, 2 >(),
          R"doc(
            Minimal residual iterative solver

            Args:
                operator:        operator to invert
                preconditioner:  preconditioner to use (i.e., apprixmate inverse of the operator)
                reduction:       factor to reduce the defect by
                maxIterations:   maximum number of iterations to perform
                verbose:         verbosity level (0 = quiet, 1 = summary, 2 = verbose)

            Returns:
                ISTL Minimal residual solver

            Note:
                The minimal residual method can only be applied if the operator and the preconditioner are both symmetric.
          )doc" );
      }

      template< class X, class Y, class... options >
      inline std::enable_if_t< std::is_same< X, Y >::value >
      registerEndomorphismSolvers ( pybind11::module module, pybind11::class_< LinearOperator< X, Y >, options... >, PriorityTag< 0 > )
      {}

      template< class X, class Y, class... options >
      inline void registerEndomorphismSolvers ( pybind11::module module, pybind11::class_< LinearOperator< X, Y >, options... > cls )
      {
        registerEndomorphismSolvers( module, cls, PriorityTag< 42 >() );
      }

    } // namespace detail



    // registerSolvers
    // ---------------

    template< class X, class Y, class... options >
    inline void registerSolvers ( pybind11::module module, pybind11::class_< LinearOperator< X, Y >, options... > cls )
    {
      typedef Dune::InverseOperator< X, Y > Solver;

      using pybind11::operator""_a;

      pybind11::options opts;
      opts.disable_function_signatures();

      pybind11::class_< Solver > clsSolver( module, "InverseOperator" );
      registerInverseOperator( clsSolver );

      detail::registerEndomorphismSolvers( module, cls );

      module.def( "RestartedGMResSolver", [] ( LinearOperator< X, Y > &op, Preconditioner< X, Y > &prec, double reduction, int restart, int maxit, int verbose ) {
          return static_cast< Solver * >( new Dune::RestartedGMResSolver< X, Y >( op, prec, reduction, restart, maxit, verbose ) );
        }, "operator"_a, "preconditioner"_a, "reduction"_a, "restart"_a, "maxIterations"_a = std::numeric_limits< int >::max(), "verbose"_a = 0, pybind11::keep_alive< 0, 1 >(), pybind11::keep_alive< 0, 2 >(),
          R"doc(
            Restarted generalized minimal residual iterative solver

            Args:
                operator:        operator to invert
                preconditioner:  preconditioner to use (i.e., apprixmate inverse of the operator)
                reduction:       factor to reduce the defect by
                restart:         number of iterations before restart
                maxIterations:   maximum number of iterations to perform
                verbose:         verbosity level (0 = quiet, 1 = summary, 2 = verbose)

            Returns:
                ISTL Restarted generalized minimal residual solver

            Note:
                The restarted generalized minimal residual method holds restart many vectors in memory during application.
                This can lead to a large memory consumption.
          )doc" );
    }

  } // namespace Python

} // namespace Dune

#endif // #ifndef DUNE_PYTHON_ISTL_SOLVER_HH
