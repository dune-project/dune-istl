// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solverrepository.hh>

#include "laplacian.hh"

int main(int argc, char** argv){

  Dune::ParameterTree config;
  Dune::ParameterTreeParser::readINITree("solverrepositorytest.ini", config);
  Dune::ParameterTreeParser::readOptions(argc, argv, config);

  using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;

  Matrix mat;
  int N = config.get("problem.N", 100);
  setupLaplacian(mat, N);
  Vector x(mat.M()), b(mat.N());

  using Operator = Dune::MatrixAdapter<Matrix, Vector, Vector>;
  std::shared_ptr<Dune::LinearOperator<Vector, Vector>> op = std::make_shared<Operator>(mat);

  auto solver = Dune::SolverRepository<Dune::LinearOperator<Vector, Vector>>::get(op, config.sub("solver"));

  x = 0;
  b = 1;
  Dune::InverseOperatorResult res;
  solver->apply(x,b,res);
  return 0;
}
