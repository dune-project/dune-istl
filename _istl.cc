// -*- tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include <config.h>

#include <dune/common/fmatrix.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include <dune/corepy/istl/bcrsmatrix.hh>
#include <dune/corepy/istl/bvector.hh>

#include <dune/corepy/pybind11/pybind11.h>

PYBIND11_PLUGIN(_istl)
{
  pybind11::module module( "_istl" );

  Dune::CorePy::registerBCRSMatrix< Dune::BCRSMatrix< Dune::FieldMatrix< double, 1, 1 > > >( module );
  Dune::CorePy::registerBlockVector< Dune::BlockVector< Dune::FieldVector< double, 1 > > >( module );

  // export solver category
  pybind11::enum_< Dune::SolverCategory::Category > solverCategory( module, "SolverCategory" );
  solverCategory.value( "sequential", Dune::SolverCategory::sequential );
  solverCategory.value( "nonoverlapping", Dune::SolverCategory::nonoverlapping );
  solverCategory.value( "overlapping", Dune::SolverCategory::overlapping );

  return module.ptr();
}
