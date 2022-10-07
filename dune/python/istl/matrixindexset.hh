// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_PYTHON_ISTL_BCRSMATRIX_HH
#define DUNE_PYTHON_ISTL_BCRSMATRIX_HH

#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <dune/istl/matrixindexset.hh>

#include <dune/python/pybind11/pybind11.h>
#include <dune/python/pybind11/stl.h>

namespace Dune
{

  namespace Python
  {

    // registermatrixindexset
    // ------------------

    template <class MatrixIndexSet, class... options>
    void registerMatrixIndexSet(pybind11::handle scope,
                            pybind11::class_<MatrixIndexSet, options...> cls)
    {
      typedef std::size_t size_type;

      // two different possible constructors
      cls.def( pybind11::init( [] () { return new MatrixIndexSet(); } ) );
      cls.def( pybind11::init( [] (size_type rows, size_type cols) { return new MatrixIndexSet(rows,cols); } ) );

      cls.def ( "add", [] (MatrixIndexSet &self, size_type i, size_type j) {self.add(i,j); } );

      cls.def( "exportIdx<BCRSMatrix etc...>", [] (MatrixIndexSet &self) { } );
    }

    template< class MatrixIndexSet >
    pybind11::class_< MatrixIndexSet > registerMatrixIndexSet ( pybind11::handle scope, const char *clsName = "MatrixIndexSet" )
    {
      pybind11::class_< MatrixIndexSet > cls( scope, clsName );
      registerMatrixIndexSet( scope, cls );
      return cls;
    }

  } // namespace Python

} // namespace Dune

#endif // #ifndef DUNE_PYTHON_ISTL_BCRSMATRIX_HH
