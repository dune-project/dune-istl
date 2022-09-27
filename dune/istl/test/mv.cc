// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/timer.hh>
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

  int iter=1000;

  for(int i=0; i< iter; ++i)
    fop.apply(x,b);

  std::cout<<iter<<" MV took "<<watch.elapsed()<<" seconds"<<std::endl;
  watch.reset();

  for(int i=0; i< iter; ++i) {
    x=0;
    mat.umv(b,x);
  }

  std::cout<<iter<<" equivalent UMV took "<<watch.elapsed()<<" seconds"<<std::endl;
}
