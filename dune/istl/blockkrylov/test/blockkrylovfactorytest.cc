#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <dune/common/simd/loop.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/istl/solverfactory.hh>
#include <dune/istl/blockkrylov/blockcg.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/test/laplacian.hh>

using namespace Dune;

const int BS=1;
typedef FieldMatrix<double,BS,BS> MatrixBlock;
typedef BCRSMatrix<MatrixBlock> BCRSMat;
typedef LoopSIMD<double, 16, 32> SIMD;
typedef FieldVector<SIMD,BS> VectorBlock;
typedef BlockVector<VectorBlock> BVector;
typedef MatrixAdapter<BCRSMat,BVector,BVector> Operator;

std::shared_ptr<Operator> fop;
int N=50;

TestSuite test(const ParameterTree& config){
  TestSuite tsuite;
  BVector b(N*N), x(N*N);
  fillRandom(x, SIMD(0.0)==0.0);
  fop->apply(x, b);
  BVector b0 = b;
  x=0;
  auto solver = getSolverFromFactory(fop, config);
  InverseOperatorResult res;
  solver->apply(x,b, res);

  // compute residual
  auto def0 = b0.two_norm();
  fop->applyscaleadd(-1.0, x, b0);
  tsuite.check(Simd::allTrue(100 * b0.two_norm() > config.get<double>("reduction")*def0), "convergence test failed!");
  return tsuite;
}

int main(int argc, char** argv)
{
  TestSuite tsuite;

  ParameterTree ptree;
  ParameterTreeParser::readINITree("blockkrylovfactorytest.ini", ptree, true);

  Dune::initSolverFactories<Operator>();
  BCRSMat mat;
  fop = std::make_shared<Operator>(mat);

  setupLaplacian(mat,N);

  for(const auto& subkey : ptree.getSubKeys()){
    std::cout << "[" << subkey << "]" << std::endl;
    tsuite.subTest(test(ptree.sub(subkey)));
  }
  return tsuite.exit();
}
