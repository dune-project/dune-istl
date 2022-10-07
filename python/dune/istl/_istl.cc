// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include <config.h>

#include <dune/common/fmatrix.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include <dune/python/common/fvector.hh>
#include <dune/python/istl/bcrsmatrix.hh>
#include <dune/python/istl/bvector.hh>
#include <dune/python/istl/operators.hh>
#include <dune/python/istl/preconditioners.hh>
#include <dune/python/istl/solvers.hh>

#include <dune/python/pybind11/pybind11.h>

#include <dune/python/common/typeregistry.hh>

PYBIND11_MODULE( _istl, module )
{
  // export solver category
  pybind11::enum_< Dune::SolverCategory::Category > solverCategory( module, "SolverCategory" );
  solverCategory.value( "sequential", Dune::SolverCategory::sequential );
  solverCategory.value( "nonoverlapping", Dune::SolverCategory::nonoverlapping );
  solverCategory.value( "overlapping", Dune::SolverCategory::overlapping );

  // export block vector with block size 1
  typedef Dune::BlockVector< Dune::FieldVector< double, 1 > > Vector;
  Dune::Python::registerBlockVector< Vector >( module );

  // export linear operator, preconditioners, and solvers for blockvectors with block size 1
  pybind11::class_< Dune::LinearOperator< Vector, Vector > > clsLinearOperator( module, "LinearOperator" );
  Dune::Python::registerLinearOperator( clsLinearOperator );
  Dune::Python::registerPreconditioners( module, clsLinearOperator );
  Dune::Python::registerSolvers( module, clsLinearOperator );

  // export BCRS matrix with block size 1x1
  typedef Dune::BCRSMatrix< Dune::FieldMatrix< double, 1, 1 > > Matrix;

  // register the BCRS matrix of size 1-1 in the type registry
  Dune::Python::registerBCRSMatrix< Matrix >( module );

  // export buildmode
  using BuildMode = Matrix::BuildMode;
  pybind11::enum_< BuildMode > buildMode( module, "BuildMode" );
  buildMode.value( "row_wise", BuildMode::row_wise );
  buildMode.value( "random",   BuildMode::random );
  buildMode.value( "implicit", BuildMode::implicit );
  buildMode.value( "unknown",  BuildMode::unknown );


  // export matrix-based preconditioners for BCRS matrix with block size 1x1
  Dune::Python::registerMatrixPreconditioners< Matrix >( module, clsLinearOperator );
}
