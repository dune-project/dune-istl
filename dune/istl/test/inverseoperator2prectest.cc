// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include<dune/istl/bvector.hh>
#include<dune/istl/superlu.hh>
#include<dune/istl/preconditioners.hh>
#include<dune/istl/solvers.hh>
#include<dune/istl/operators.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include "laplacian.hh"

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

  Dune::InverseOperatorResult res, res1;
  x=1;
  mat.mv(x, b);
  x=0;
  Dune::SeqJac<BCRSMat,BVector,BVector> prec0(mat, 1,1.0);
  Dune::LoopSolver<BVector> solver0(fop, prec0, 1e-3,10,0);
  Dune::InverseOperator2Preconditioner<Dune::LoopSolver<BVector>>
    prec(solver0);
  Dune::LoopSolver<BVector> solver(fop, prec, 1e-8,10,2);
  solver.apply(x,b,res);

  x=1;
  mat.mv(x, b);
  x=0;
  std::cout<<"solver1"<<std::endl;
  Dune::LoopSolver<BVector> solver1(fop, prec0, 1e-8,100,2);
  solver1.apply(x,b,res1);

  if(res1.iterations!=res.iterations*10)
  {
    std::cerr<<"Convergence rates do not match!"<<std::endl;
    return 1;
  }
  return 0;
}
