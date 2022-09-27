// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_PYTHON_ISTL_PRECONDITIONER_HH
#define DUNE_PYTHON_ISTL_PRECONDITIONER_HH

#include <dune/common/typeutilities.hh>
#include <dune/common/version.hh>

#include <dune/istl/operators.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/preconditioners.hh>

#include <dune/python/pybind11/pybind11.h>

namespace Dune
{

  namespace Python
  {

    // registerPreconditioner
    // ----------------------

    template< class Preconditioner, class... options >
    inline void registerPreconditioner ( pybind11::class_< Preconditioner, options... > cls )
    {
      typedef typename Preconditioner::domain_type domain_type;
      typedef typename Preconditioner::range_type range_type;

      using pybind11::operator""_a;

      pybind11::options opts;
      opts.disable_function_signatures();

      cls.def( "__call__", [] ( Preconditioner &self, domain_type &v, const range_type &d ) {
          self.apply( v, d );
        }, "update", "defect" );

      cls.def( "pre", [] ( Preconditioner &self, domain_type &x, range_type &b ) {
          self.pre( x, b );
        }, "x"_a, "rhs"_a );

      cls.def( "post", [] ( Preconditioner &self, domain_type &x ) {
          self.post( x );
        }, "x"_a );

      cls.def_property_readonly( "category", [] ( const Preconditioner &self ) { return self.category(); } );
    }



    // registerPreconditioners
    // -----------------------

    template< class X, class Y, class... options >
    inline void registerPreconditioners ( pybind11::module module, pybind11::class_< LinearOperator< X, Y >, options...  > cls )
    {
      typedef Dune::Preconditioner< X, Y > Preconditioner;

      typedef typename Preconditioner::field_type field_type;

      using pybind11::operator""_a;

      pybind11::class_< Preconditioner > clsPreconditioner( module, "Preconditioner" );
      registerPreconditioner( clsPreconditioner );

      module.def( "Richardson", [] ( field_type w ) {
          return static_cast< Preconditioner * >( new Richardson< X, Y >( w ) );
        }, "relaxation"_a = field_type( 1 ),
        R"doc(
          Richardson preconditioner

          Args:
              relaxation:  factor to relax the input by (default: 1)

          Returns:
              ISTL Richardson preconditioner

          Note:
              Use this preconditioner with default parameters if you do not want to apply preconditioning.
              This is a sequential preconditioner.
        )doc" );
    }

    template< class M, class X, class Y, class... options >
    inline void registerMatrixPreconditioners ( pybind11::module module, pybind11::class_< LinearOperator< X, Y >, options... > cls )
    {
      typedef Dune::Preconditioner< X, Y > Preconditioner;

      typedef typename Preconditioner::field_type field_type;

      using pybind11::operator""_a;

      pybind11::options opts;
      opts.disable_function_signatures();

      module.def( "SeqSSOR", [] ( const M &A, int n, field_type w ) {
          return static_cast< Preconditioner * >( new SeqSSOR< M, X, Y >( A, n, w ) );
        }, "matrix"_a, "iterations"_a = 1, "relaxation"_a = field_type( 1 ),
        R"doc(
          Symmetric successive over-relaxation preconditioner

          Args:
              matrix:      matrix to precondition
              iterations:  number of iterations to perform
              relaxation:  factor to relax the iterations

          Returns:
              ISTL Sequential symmetric successive over-relaxation preconditioner

          Note:
              The symmetric successive over-relaxation iteration can only be applied if the matrix is symmetric and the diagonal entries are all non-zero.
        )doc" );

      module.def( "SeqSOR", [] ( const M &A, int n, field_type w ) {
          return static_cast< Preconditioner * >( new SeqSOR< M, X, Y >( A, n, w ) );
        }, "matrix"_a, "iterations"_a = 1, "relaxation"_a = field_type( 1 ),
        R"doc(
          Successive over-relaxation preconditioner

          Args:
              matrix:      matrix to precondition
              iterations:  number of iterations to perform
              relaxation:  factor to relax the iterations (default: 1)

          Returns:
              ISTL Sequential successive over-relaxation preconditioner

          Note:
            The successive over-relaxation iteration can only be applied if the matrix diagonal entries are all non-zero.
        )doc" );

      module.def( "SeqGaussSeidel", [] ( const M &A, int n, field_type w ) {
          return static_cast< Preconditioner * >( new SeqGS< M, X, Y >( A, n, w ) );
        }, "matrix"_a, "iterations"_a = 1, "relaxation"_a = field_type( 1 ),
        R"doc(
          Gauss-Seidel preconditioner

          Args:
              matrix:      matrix to precondition
              iterations:  number of iterations to perform
              relaxation:  factor to relax the iterations (default: 1)

          Returns:
              ISTL Sequential Gauss-Seidel preconditioner

          Note:
              The Gauss-Seidel iteration can only be applied if the matrix diagonal entries are all non-zeros.
        )doc" );

      module.def( "SeqJacobi", [] ( const M &A, int n, field_type w ) {
          return static_cast< Preconditioner * >( new SeqJac< M, X, Y >( A, n, w ) );
        }, "matrix"_a, "iterations"_a = 1, "relaxation"_a = field_type( 1 ),
        R"doc(
          Jacobi preconditioner

          Args:
              matrix:      matrix to precondition
              iterations:  number of iterations to perform
              relaxation:  factor to relax the iterations (default: 1)

          Returns:
              ISTL Sequential Jacobi preconditioner

          Note:
              The Jacobi iteration can only be applied if the matrix diagonal entries are all non-zeros.
        )doc" );

      module.def( "SeqILU", [] ( const M &A, int n, field_type w ) {
          return static_cast< Preconditioner * >( new SeqILU< M, X, Y >( A, n, w ) );
        }, "matrix"_a, "iterations"_a = 1, "relaxation"_a = field_type( 1 ),
        R"doc(Incomplete LU factorization (with fill-in) preconditioner

          Args:
              matrix:      matrix to precondition
              iterations:  number of fill-in iterations
              relaxation:  factor to relax the iterations (default: 1)

          Returns:
              ISTL Sequential incomplete LU factorization preconditioner

          Note:
              The incomplete LU factorization with fill-in has the same sparsity pattern as the given matrix.
        )doc" );

      module.def( "SeqILDL", [] ( const M &A, field_type w ) {
          return static_cast< Preconditioner * >( new SeqILDL< M, X, Y >( A, w ) );
        }, "matrix"_a, "relaxation"_a = field_type( 1 ),
        R"doc(
          Incomplete LDL factorization preconditioner

          Args:
              matrix:      matrix to precondition
              relaxation:  factor to relax the iterations (default: 1)

          Returns:
              ISTL Sequential incomplete LDL factorization preconditioner

          Note:
              The matrix is assumed to by symmetric, so the upper triangular matrix is ignored
              and need not be assembled.
              The incomplete LDL factorization with fill-in has the same sparsity pattern as the
              given matrix, however only L and D are actually stored.
        )doc" );
    }

  } // namespace Python

} // namespace Dune

#endif // #ifndef DUNE_PYTHON_ISTL_PRECONDITIONER_HH
