#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/simd/loop.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/istl/solverfactory.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/blockkrylov/blockcg.hh>
#include <dune/istl/blockkrylov/blockgmres.hh>
#include <dune/istl/blockkrylov/blockbicgstab.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/test/laplacian.hh>
#include <dune/istl/paamg/test/anisotropic.hh>

using namespace Dune;

const int BS=1;
typedef FieldMatrix<double,BS,BS> MatrixBlock;
typedef BCRSMatrix<MatrixBlock> BCRSMat;
typedef LoopSIMD<double, 16, 32> SIMD;
typedef FieldVector<SIMD,BS> VectorBlock;
typedef BlockVector<VectorBlock> BVector;
#if HAVE_MPI
typedef OwnerOverlapCopyCommunication<size_t> Comm;
typedef OverlappingSchwarzOperator<BCRSMat,BVector,BVector,Comm> Operator;
#else
typedef MatrixAdapter<BCRSMat,BVector,BVector> Operator;
#endif

std::shared_ptr<Operator> fop;
int N=50;

TestSuite test(const ParameterTree& config){
  TestSuite tsuite;
  BVector b(fop->getmat().N()), x(fop->getmat().M());
  std::srand(42);
  fillRandom(x, SIMD(0.0)==0.0);
  fop->apply(x, b);
  BVector b0 = b;
  x=0;
  auto solver = getSolverFromFactory(fop, config);
  InverseOperatorResult res;
  solver->apply(x,b, res);

  // compute residual
#if HAVE_MPI
  Comm comm(MPI_COMM_WORLD);
  ParallelScalarProduct<BVector,Comm> sp(comm, SolverCategory::overlapping);
  double reduction_factor = comm.communicator().size()*200.0;
#else
  ScalarProduct<BVector> sp;
  double reduction_factor = 100.0;
#endif
  auto def0 = sp.norm(b0);
  fop->applyscaleadd(-1.0, x, b0);
  auto reduction = sp.norm(b0)/def0;
  tsuite.check(Simd::allTrue(reduction < config.get<double>("reduction")), "convergence test failed!");
  return tsuite;
}

int main(int argc, char** argv)
{
  auto& mpihelper = MPIHelper::instance(argc, argv);

  TestSuite tsuite;

  ParameterTree ptree;
  ParameterTreeParser::readINITree("blockkrylovfactorytest.ini", ptree, true);

  N = ptree.get("N", 50);

  Dune::initSolverFactories<Operator>();
#if HAVE_MPI
  Comm comm(MPI_COMM_WORLD);
  int n;
  BCRSMat mat = setupAnisotropic2d<MatrixBlock>(N, comm.indexSet(), comm.communicator(), &n);
  comm.remoteIndices().template rebuild<false>();
  fop = std::make_shared<Operator>(mat,comm);
#else
  BCRSMat mat;
  setupLaplacian(mat,N);
  fop = std::make_shared<Operator>(mat);
#endif


  for(const auto& subkey : ptree.getSubKeys()){
    if(mpihelper.rank() == 0){
      std::cout << std::endl;
      std::cout << "[" << subkey << "]" << std::endl;
    }else
      ptree.sub(subkey)["verbose"] = "0";
    tsuite.subTest(test(ptree.sub(subkey)));
  }
  return tsuite.exit();
}
