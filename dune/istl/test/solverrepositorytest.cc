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

// include solvers, this should later be done in a dedicated library
#include <dune/istl/umfpack.hh>
#include <dune/istl/superlu.hh>

#include "laplacian.hh"

using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;
using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;

template<class Comm>
void testSeq(const Dune::ParameterTree& config, Comm c){
  if(c.rank() == 0){
    Matrix mat;
    int N = config.get("N", 10);
    setupLaplacian(mat, N);
    Vector x(mat.M()), b(mat.N());

    using Operator = Dune::MatrixAdapter<Matrix, Vector, Vector>;
    std::shared_ptr<Operator> op = std::make_shared<Operator>(mat);

    for(std::string test : config.getSubKeys()){
      Dune::ParameterTree solverConfig = config.sub(test);
      std::cout << " ============== " << test << " ============== " << std::endl;
      std::shared_ptr<Dune::InverseOperator<Vector, Vector>> solver = getSolverFromRepository(op, solverConfig);
      x = 0;
      b = 1;
      Dune::InverseOperatorResult res;
      solver->apply(x,b,res);
    }
  }
}

// template<class Comm>
// void testOverlapping(const Dune::ParameterTree& config, Comm c){
//   Matrix mat;
//   int N = config.get("N", 10);
//   typedef Dune::OwnerOverlapCopyCommunication<int> Communication;
//   Communication comm(c);
//   int n;
//   mat = setupAnisotropic2d<Dune::FieldMatrix<double, 1, 1>>(N, comm.indexSet(), comm.communicator(), &n);
//   comm.remoteIndices().template rebuild<false>();
//   Vector x(mat.M()), b(mat.N());

//   using Operator = Dune::OverlappingSchwarzOperator<Matrix, Vector, Vector, Communication>;
//   std::shared_ptr<Operator> op = std::make_shared<Operator>(mat, comm);

//   for(const std::string& test : config.getSubKeys()){
//     Dune::ParameterTree solverConfig = config.sub(test);
//     if(c.rank() == 0)
//       std::cout << " ============== " << test << " ============== " << std::endl;
//     std::shared_ptr<Dune::InverseOperator<Vector, Vector>> solver = getSolverFromRepository(op, solverConfig);
//     x = 1;
//     b = 0;
//     setBoundary(x, b, N, comm.indexSet());
//     Dune::InverseOperatorResult res;
//     solver->apply(x,b,res);
//   }
// }

// template<class Comm>
// void testNonoverlapping(const Dune::ParameterTree& config, Comm c){
//   Matrix mat;
//   int N = config.get("N", 10);
//   typedef Dune::OwnerOverlapCopyCommunication<int> Communication;
//   Communication comm(c, Dune::SolverCategory::nonoverlapping);
//   int n;
//   mat = setupAnisotropic2d<Dune::FieldMatrix<double, 1, 1>>(N, comm.indexSet(), comm.communicator(), &n);
//   comm.remoteIndices().template rebuild<false>();
//   Vector x(mat.M()), b(mat.N());

//   using Operator = Dune::NonoverlappingSchwarzOperator<Matrix, Vector, Vector, Communication>;
//   std::shared_ptr<Operator> op = std::make_shared<Operator>(mat, comm);

//   for(const std::string& test : config.getSubKeys()){
//     Dune::ParameterTree solverConfig = config.sub(test);
//     if(c.rank() == 0)
//       std::cout << " ============== " << test << " ============== " << std::endl;
//     std::shared_ptr<Dune::InverseOperator<Vector, Vector>> solver = getSolverFromRepository(op, solverConfig);
//     x = 1;
//     b = 0;
//     setBoundary(x, b, N, comm.indexSet());
//     Dune::InverseOperatorResult res;
//     solver->apply(x,b,res);
//   }
// }

int main(int argc, char** argv){
  auto& mpihelper = Dune::MPIHelper::instance(argc, argv);
  Dune::ParameterTree config;
  Dune::ParameterTreeParser::readINITree("solverrepositorytest.ini", config);
  Dune::ParameterTreeParser::readOptions(argc, argv, config);

  std::cout << std::endl << " Testing sequential tests... " << std::endl;
  testSeq(config.sub("sequential"), mpihelper.getCollectiveCommunication());
  // std::cout << std::endl << " Testing overlapping tests... " << std::endl;
  // testOverlapping(config.sub("overlapping"), mpihelper.getCollectiveCommunication());
  //   std::cout << std::endl << " Testing nonoverlapping tests... " << std::endl;
  // testNonoverlapping(config.sub("nonoverlapping"), mpihelper.getCollectiveCommunication());
  return 0;
}
