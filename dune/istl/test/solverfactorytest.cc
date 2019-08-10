// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/solverfactoryregisterhelpers.hh>

#include "laplacian.hh"

int main(int argc, char** argv){
  auto& mpihelper = Dune::MPIHelper::instance(argc, argv);

  Dune::TestSuite testSuite;

  //setup problem
  typedef Dune::BlockVector<Dune::FieldVector<double, 1>> Vector;
  typedef Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>> Matrix;
  Matrix mat;

  Dune::ParameterTree config;
  Dune::ParameterTreeParser::readINITree("solverfactorytest.ini", config);
  Dune::ParameterTreeParser::readOptions(argc, argv, config);

  int N = config.get("problem.N", 100);
  setupLaplacian(mat, N);
  Vector x(mat.M()), b(mat.N());
  Dune::InverseOperatorResult res;

  std::shared_ptr<Dune::LinearOperator<Vector, Vector>> op = std::make_shared<Dune::MatrixAdapter<Matrix, Vector, Vector>>(mat);

  Dune::registerISTLIterativeSolvers<Vector>();
  Dune::registerISTLDirectSolvers<Matrix>();

  int counter = 1;
  while(config.hasSub(std::string("solver")+std::to_string(counter))){
    Dune::ParameterTree solverconfig = config.sub(std::string("solver")+std::to_string(counter));
    std::shared_ptr<Dune::InverseOperator<Vector, Vector>> solver = Dune::SolverFactory<Vector, Vector>::getSolver(op, solverconfig);
    x = 0;
    b = 1;
    solver->apply(x,b,res);
  }
  return testSuite.exit();
}
