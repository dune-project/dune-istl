// -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include <config.h>

#include <dune/common/fmatrix.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include <dune/corepy/istl/bcrsmatrix.hh>
#include <dune/corepy/istl/bvector.hh>
#include <dune/corepy/istl/operators.hh>
#include <dune/corepy/istl/preconditioners.hh>
#include <dune/corepy/istl/solvers.hh>

#include <dune/corepy/pybind11/pybind11.h>

PYBIND11_PLUGIN(_istl)
{
  pybind11::module module( "_istl" );

  // export solver category
  pybind11::enum_< Dune::SolverCategory::Category > solverCategory( module, "SolverCategory" );
  solverCategory.value( "sequential", Dune::SolverCategory::sequential );
  solverCategory.value( "nonoverlapping", Dune::SolverCategory::nonoverlapping );
  solverCategory.value( "overlapping", Dune::SolverCategory::overlapping );

  // export block vector with block size 1
  typedef Dune::BlockVector< Dune::FieldVector< double, 1 > > Vector;
  Dune::CorePy::registerBlockVector< Vector >( module );

  // export linear operator, preconditioners, and solvers for blockvectors with block size 1
  pybind11::class_< Dune::LinearOperator< Vector, Vector > > clsLinearOperator( module, "LinearOperator" );
  Dune::CorePy::registerLinearOperator( clsLinearOperator );
  Dune::CorePy::registerPreconditioners( module, clsLinearOperator );
  Dune::CorePy::registerSolvers( module, clsLinearOperator );

  // export BCRS matrix with block size 1x1
  typedef Dune::BCRSMatrix< Dune::FieldMatrix< double, 1, 1 > > Matrix;
  Dune::CorePy::registerBCRSMatrix< Matrix >( module );

  // export matrix-based preconditioners for BCRS matrix with block size 1x1
  Dune::CorePy::registerMatrixPreconditioners< Matrix >( module, clsLinearOperator );

  return module.ptr();
}
