// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
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

#include <complex>
#include <iterator>

namespace Dune
{
  using Vec1 = BlockVector<FieldVector<double,1>>;
  using Vec2 = BlockVector<FieldVector<std::complex<double>,1>>;

  // explicit template instantiation of all iterative solvers

  // field_type = double
  template class InverseOperator<Vec1,Vec1>;
  template class LoopSolver<Vec1>;
  template class GradientSolver<Vec1>;
  template class CGSolver<Vec1>;
  template class BiCGSTABSolver<Vec1>;
  template class MINRESSolver<Vec1>;
  template class RestartedGMResSolver<Vec1>;
  template class RestartedFlexibleGMResSolver<Vec1>;
  template class GeneralizedPCGSolver<Vec1>;
  template class RestartedFCGSolver<Vec1>;
  template class CompleteFCGSolver<Vec1>;

  // field_type = complex<double>
  template class InverseOperator<Vec2,Vec2>;
  template class LoopSolver<Vec2>;
  template class GradientSolver<Vec2>;
  template class CGSolver<Vec2>;
  template class BiCGSTABSolver<Vec2>;
  template class MINRESSolver<Vec2>;
  template class RestartedGMResSolver<Vec2>;
  template class RestartedFlexibleGMResSolver<Vec2>;
  template class GeneralizedPCGSolver<Vec2>;
  template class RestartedFCGSolver<Vec2>;
  template class CompleteFCGSolver<Vec2>;

} // end namespace Dune


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
  const Operator fop(mat);
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

  Dune::RestartedGMResSolver<BVector> solver3(fop, prec0, 1e-3,5,20,2);
  solver3.apply(x,b, res);

  b = 0;
  x = 1;
  mat.mv(x, b);
  x = 0;

  Dune::RestartedFCGSolver<BVector> solver4(fop, prec0, 1e-3,10,2);
  solver4.apply(x, b, res);

  b = 0;
  x = 1;
  mat.mv(x, b);
  x = 0;

  Dune::CompleteFCGSolver<BVector> solver5(fop, prec0, 1e-3,10,2);
  solver5.apply(x, b, res);

  b = 0;
  x = 1;
  mat.mv(x, b);
  x = 99;

  Dune::RestartedFlexibleGMResSolver<BVector> solver6(fop, prec0, 1e-3, 5, 20, 2);
  solver6.apply(x,b, res);

  return 0;
}
