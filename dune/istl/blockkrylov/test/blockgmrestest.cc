#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <dune/common/simd/loop.hh>
#include <dune/common/test/testsuite.hh>

#include <dune/istl/blockkrylov/blockgmres.hh>
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
std::shared_ptr<Preconditioner<BVector,BVector>> prec;
int N=50;

template<size_t P>
TestSuite test(){
  TestSuite tsuite;
  BVector b(N*N), x(N*N);
  fillRandom(x, SIMD(0.0)==0.0);
  fop->apply(x, b);
  BVector b0 = b;
  x=0;
  ParameterTree config;
  config["reduction"] = "1e-14";
  config["maxit"] = "1000";
  config["verbose"] = "1";
  BlockGMRes<BVector, BVector, P, false> solver0(fop, prec, config);
  InverseOperatorResult res;
  solver0.apply(x,b, res);

  // compute residual
  auto def0 = b0.two_norm();
  tsuite.check(Simd::allTrue(100 * b0.two_norm() > config.get<double>("reduction")*def0), "convergence test failed!");
  return tsuite;
}

int main(int argc, char** argv)
{
  TestSuite tsuite;
  if(argc>1)
    N = atoi(argv[1]);
  std::cout<<"testing for N="<<N<<" BS="<<1<<std::endl;
  std::cout << "S=" << Simd::lanes<SIMD>() << " RHS" << std::endl;

  BCRSMat mat;
  fop = std::make_shared<Operator>(mat);

  setupLaplacian(mat,N);
  prec = std::make_shared<SeqILU<BCRSMat,BVector,BVector>>(mat, 0,1.0);// nonsymmetric preconditioner

  Hybrid::forEach(std::index_sequence<1, 2, 4, 8, 16>{}, [&tsuite](auto p){
    std::cout << "P=" << size_t(p) << std::endl;
    tsuite.subTest(test<p.value>());
  });
  return tsuite.exit();
}
