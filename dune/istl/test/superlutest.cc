// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include <complex>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/timer.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/superlu.hh>

#include "laplacian.hh"

#ifndef SUPERLU_NTYPE
#define SUPERLU_NTYPE 1
#endif

#if SUPERLU_NTYPE==1
typedef double FIELD_TYPE;
#endif

#if SUPERLU_NTYPE==0
typedef float FIELD_TYPE;
#endif

#if SUPERLU_NTYPE==2
typedef std::complex<float> FIELD_TYPE;
#endif

#if SUPERLU_NTYPE>=3
typedef std::complex<double> FIELD_TYPE;
#endif

int main(int argc, char** argv)
{
  const int BS=1;
  std::size_t N=100;

  if(argc>1)
    N = atoi(argv[1]);
  std::cout<<"testing for N="<<N<<" BS="<<1<<std::endl;


  typedef Dune::FieldMatrix<FIELD_TYPE,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<FIELD_TYPE,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> Vector;
  typedef Dune::MatrixAdapter<BCRSMat,Vector,Vector> Operator;

  BCRSMat mat;
  Operator fop(mat);
  Vector b(N*N), x(N*N);

  setupLaplacian(mat,N);
  b=1;
  x=0;

  Dune::Timer watch;

  watch.reset();

  Dune::SuperLU<BCRSMat> solver(mat, true);

  Dune::InverseOperatorResult res;

  Dune::SuperLU<BCRSMat> solver1;

  std::set<std::size_t> mrs;
  for(std::size_t s=0; s < N/2; ++s)
    mrs.insert(s);

  solver1.setSubMatrix(mat,mrs);
  solver.setVerbosity(true);
  solver.apply(x,b, res);

  std::cout<<"Defect reduction is "<<res.reduction<<std::endl;
  solver1.apply(x,b, res);
  solver1.apply(reinterpret_cast<FIELD_TYPE*>(&x[0]), reinterpret_cast<double*>(&b[0]));
}
