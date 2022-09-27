// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_PYTHON_ISTL_OPERATORS_HH
#define DUNE_PYTHON_ISTL_OPERATORS_HH

#include <dune/common/typeutilities.hh>

#include <dune/istl/operators.hh>

#include <dune/python/pybind11/pybind11.h>

namespace Dune
{

  namespace Python
  {

    // registerLinearOperator
    // ----------------------

    template< class LinearOperator, class... options >
    inline static void registerLinearOperator ( pybind11::class_< LinearOperator, options... > cls )
    {
      typedef typename LinearOperator::field_type field_type;
      typedef typename LinearOperator::domain_type domain_type;
      typedef typename LinearOperator::range_type range_type;

      using pybind11::operator""_a;

      // application
      cls.def( "apply", [] ( const LinearOperator &self, const domain_type &x, range_type &y ) {
          self.apply( x, y );
        }, "x"_a, "y"_a );
      cls.def( "applyscaleadd", [] ( const LinearOperator &self, const field_type &alpha, const domain_type &x, range_type &y ) {
          self.applyscaleadd( alpha, x, y );
        }, "alpha"_a, "x"_a, "y"_a );

      // linear operator
      cls.def( "asLinearOperator", [] ( pybind11::object self ) { return self; } );
    }

  } // namespace Python

} // namespace Dune

#endif // #ifndef DUNE_PYTHON_ISTL_BCRSMATRIX_HH
