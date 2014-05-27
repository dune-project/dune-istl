// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/io.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/timer.hh>
#include <dune/istl/overlappingschwarz.hh>
#include <dune/istl/solvers.hh>
#include "laplacian.hh"

#include <iterator>

int main(int argc, char** argv)
{

  const int BS=1;
  int N=100;

  if(argc>1)
    N = atoi(argv[1]);
  std::cout<<"testing for N="<<N<<" BS="<<1<<std::endl;


  typedef Dune::FieldMatrix<double,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<double,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> BVector;
  typedef Dune::MatrixAdapter<BCRSMat,BVector,BVector> Operator;

  BCRSMat mat;
  Operator fop(mat);
  BVector b(N*N), x(N*N);

  setupLaplacian(mat,N);
  b=0;
  x=100;
  Dune::Timer watch;

  watch.reset();

  Dune::InverseOperatorResult res;
  x=1;
  mat.mv(x, b);
  x=0;
  Dune::SeqJac<BCRSMat,BVector,BVector> prec0(mat, 1,1.0);
  Dune::GeneralizedPCGSolver<BVector> solver0(fop, prec0, 1e-3,10,2);
  solver0.apply(x,b, res);

  b=0;
  x=1;
  mat.mv(x, b);
  x=0;

  Dune::CGSolver<BVector> solver1(fop, prec0, 1e-3,10,2);
  solver1.apply(x,b, res);

  b=0;
  x=1;
  mat.mv(x, b);
  x=99;

  Dune::BiCGSTABSolver<BVector> solver2(fop, prec0, 1e-3,10,2);
  solver2.apply(x,b, res);

  b=0;
  x=1;
  mat.mv(x, b);
  x=99;

  Dune::RestartedGMResSolver<BVector> solver3(fop, prec0, 1e-3,5,20,2,false);
  solver3.apply(x,b, res);

  return 0;
}
