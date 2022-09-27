// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include <config.h>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parallel/mpicommunication.hh>
#include <dune/istl/paamg/matrixhierarchy.hh>
#include <dune/istl/paamg/smoother.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/schwarz.hh>
#include "anisotropic.hh"

template<int blockSize>
void testHierarchy(int N)
{
  typedef int LocalId;
  typedef int GlobalId;
  typedef Dune::OwnerOverlapCopyCommunication<LocalId,GlobalId> Communication;
  typedef Communication::ParallelIndexSet ParallelIndexSet;
  typedef Dune::FieldMatrix<double,blockSize,blockSize> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<double,blockSize> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> Vector;

  int n;
  Communication pinfo(Dune::MPIHelper::getCommunicator());
  ParallelIndexSet& indices = pinfo.indexSet();

  typedef Dune::RemoteIndices<ParallelIndexSet> RemoteIndices;
  RemoteIndices& remoteIndices = pinfo.remoteIndices();

  BCRSMat mat = setupAnisotropic2d<MatrixBlock>(N, indices, Dune::MPIHelper::getCommunication(), &n);
  Vector b(indices.size());

  remoteIndices.rebuild<false>();

  typedef Dune::NegateSet<Communication::OwnerSet> OverlapFlags;
  typedef Dune::OverlappingSchwarzOperator<BCRSMat,Vector,Vector,Communication> Operator;
  typedef Dune::Amg::MatrixHierarchy<Operator,Communication> Hierarchy;
  typedef Dune::Amg::Hierarchy<Vector> VHierarchy;

  Operator op(mat, pinfo);
  Hierarchy hierarchy(
    Dune::stackobject_to_shared_ptr(op),
    Dune::stackobject_to_shared_ptr(pinfo));
  VHierarchy vh(
    Dune::stackobject_to_shared_ptr(b));

  typedef Dune::Amg::CoarsenCriterion<Dune::Amg::SymmetricCriterion<BCRSMat,Dune::Amg::FirstDiagonal> >
  Criterion;

  Criterion criterion(100,4);
  Dune::Timer timer;

  hierarchy.template build<OverlapFlags>(criterion);
  hierarchy.coarsenVector(vh);

  std::cout<<"Building hierarchy took "<<timer.elapsed()<<std::endl;

  std::cout<<"=== Vector hierarchy has "<<vh.levels()<<" levels! ==="<<std::endl;
  timer.reset();

  hierarchy.recalculateGalerkin(OverlapFlags());

  std::cout<<"Recalculation took "<<timer.elapsed()<<std::endl;
  std::vector<std::size_t> data;

  hierarchy.getCoarsestAggregatesOnFinest(data);
}


int main(int argc, char** argv)
{
  Dune::MPIHelper::instance(argc, argv);

  constexpr int blockSize = 1;
  int N=10;

  if(argc>1)
    N = atoi(argv[1]);

  testHierarchy<blockSize>(N);
}
