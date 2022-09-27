// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include <memory>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/matrixredistribute.hh>
#include <dune/istl/repartition.hh>
#include <dune/istl/paamg/amg.hh>

typedef double RT;

using namespace Dune;

typedef BCRSMatrix<FieldMatrix<RT,1,1>> BCRSMat;
typedef std::size_t GlobalId;
typedef OwnerOverlapCopyCommunication<GlobalId> Comm;
typedef RedistributeInformation<Comm> RedistInfo;

std::string matrixfile = "gr_30_30";

void loadMatrix(std::shared_ptr<BCRSMat>& pA){
  pA = std::make_shared<BCRSMat>();
  if(MPIHelper::getCommunication().rank() == 0){
    Dune::loadMatrixMarket(*pA, matrixfile);
  }
}

std::shared_ptr<Comm> repartMatrix(const std::shared_ptr<BCRSMat>& pA_orig, std::shared_ptr<BCRSMat>& pA, MPI_Comm comm, int size){
  typedef typename Dune::Amg::MatrixGraph<BCRSMat> MatrixGraph;
  RedistInfo ri;
  std::shared_ptr<Comm> pComm;
  Comm oocomm(comm);
  Dune::graphRepartition(MatrixGraph(*pA_orig), oocomm,
                         size,
                         pComm, ri.getInterface(), 1);
  ri.setSetup();
  pA = std::make_shared<BCRSMat>();
  redistributeMatrix(*pA_orig, *pA, oocomm, *pComm, ri);
  return pComm;
}


int main(int argc, char** argv){
  auto& mpihelper = MPIHelper::instance(argc, argv);
  auto world = mpihelper.getCommunication();

  int size = mpihelper.size();
  int rank = mpihelper.rank();
  std::shared_ptr<BCRSMat> mat, mat_reparted;
  loadMatrix(mat);
  std::shared_ptr<Comm> comm = repartMatrix(mat, mat_reparted, world, size);
  typedef FieldVector<RT,1> VectorBlock;
  typedef BlockVector<VectorBlock> Vector;
  typedef OverlappingSchwarzOperator<BCRSMat, Vector, Vector, Comm> Operator;
  std::shared_ptr<Operator> op = std::make_shared<Operator>(*mat_reparted, *comm);
  typedef Amg::CoarsenCriterion<Amg::SymmetricCriterion<BCRSMat,Amg::FirstDiagonal> >
    Criterion;
  Criterion criterion(15,200);
  typedef SeqSSOR<BCRSMat, Vector, Vector> Smoother;
  typedef BlockPreconditioner<Vector, Vector, Comm, Smoother> ParSmoother;
  typedef typename Amg::SmootherTraits<ParSmoother>::Arguments SmootherArgs;
  SmootherArgs smootherArgs;
  smootherArgs.iterations = 1;
  Amg::AMG<Operator, Vector, ParSmoother, Comm> prec(*op, criterion, smootherArgs, *comm);
  OverlappingSchwarzScalarProduct<Vector, Comm> sp(*comm);
  CGSolver<Vector> solver(*op, sp, prec, 1e-5, 300, rank==0);
  Vector x(mat_reparted->M()), b(mat_reparted->N());
  x = 1.0;
  op->apply(x, b);
  x = 0.0;
  InverseOperatorResult res;
  solver.apply(x, b, res);

  return 0;
}
