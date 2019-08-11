// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solverrepository.hh>
#include <dune/istl/paamg/test/anisotropic.hh>

#include "laplacian.hh"

using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;
using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;

template<class Comm>
void testSeq(const Dune::ParameterTree& config, Comm c){
  if(c.rank() == 0){
    Matrix mat;
    int N = config.get("problem.N", 100);
    setupLaplacian(mat, N);
    Vector x(mat.M()), b(mat.N());

    using Operator = Dune::MatrixAdapter<Matrix, Vector, Vector>;
    std::shared_ptr<Dune::LinearOperator<Vector, Vector>> op = std::make_shared<Operator>(mat);

    int counter = 1;
    while(config.hasSub(std::string("test") + std::to_string(counter))){
      Dune::ParameterTree testConfig = config.sub(std::string("test") + std::to_string(counter));
      if(testConfig.template get<std::string>("parallel", "sequential") == "sequential"){
        Dune::ParameterTree solverConfig = testConfig.sub("solver");
        std::cout << " ============== TEST " << counter << " ============== " << std::endl;
        auto solver = getSolverFromRepository(op, solverConfig);
        x = 0;
        b = 1;
        Dune::InverseOperatorResult res;
        solver->apply(x,b,res);
      }
      counter++;
    }
  }
}

template<class Comm>
void testOverlapping(const Dune::ParameterTree& config, Comm c){
  Matrix mat;
  int N = config.get("problem.N", 100);
  typedef Dune::OwnerOverlapCopyCommunication<int> Communication;
  Communication comm(c);
  int n;
  mat = setupAnisotropic2d<Dune::FieldMatrix<double, 1, 1>>(N, comm.indexSet(), comm.communicator(), &n);
  comm.remoteIndices().template rebuild<false>();
  Vector x(mat.M()), b(mat.N());

  using Operator = Dune::OverlappingSchwarzOperator<Matrix, Vector, Vector, Communication>;
  std::shared_ptr<Operator> op = std::make_shared<Operator>(mat, comm);

  int counter = 1;
  while(config.hasSub(std::string("test") + std::to_string(counter))){
    Dune::ParameterTree testConfig = config.sub(std::string("test") + std::to_string(counter));
    if(testConfig.template get<std::string>("parallel", "sequential") == "overlapping"){
      Dune::ParameterTree solverConfig = testConfig.sub("solver");
      if(c.rank() == 0)
        std::cout << " ============== TEST " << counter << " ============== " << std::endl;
      auto solver = getSolverFromRepository(op, solverConfig);
      x = 1;
      b = 0;
      setBoundary(x, b, N, comm.indexSet());
      Dune::InverseOperatorResult res;
      solver->apply(x,b,res);
    }
    counter++;
  }
}

int main(int argc, char** argv){
  auto& mpihelper = Dune::MPIHelper::instance(argc, argv);
  Dune::ParameterTree config;
  Dune::ParameterTreeParser::readINITree("solverrepositorytest.ini", config);
  Dune::ParameterTreeParser::readOptions(argc, argv, config);

  testSeq(config, mpihelper.getCollectiveCommunication());
  testOverlapping(config, mpihelper.getCollectiveCommunication());
  return 0;
}
