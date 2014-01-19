// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include <config.h>

#include "mpi.h"
#include <dune/common/parallel/mpicollectivecommunication.hh>
#include <dune/istl/paamg/hierarchy.hh>
#include <dune/istl/paamg/smoother.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/schwarz.hh>
#include "anisotropic.hh"

template<int BS>
void testHierarchy(int N)
{
  typedef int LocalId;
  typedef int GlobalId;
  typedef Dune::OwnerOverlapCopyCommunication<LocalId,GlobalId> Communication;
  typedef Communication::ParallelIndexSet ParallelIndexSet;
  typedef Dune::FieldMatrix<double,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<double,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> Vector;

  int n;
  Communication pinfo(MPI_COMM_WORLD);
  ParallelIndexSet& indices = pinfo.indexSet();

  typedef Dune::RemoteIndices<ParallelIndexSet> RemoteIndices;
  RemoteIndices& remoteIndices = pinfo.remoteIndices();

  typedef Dune::CollectiveCommunication<MPI_Comm> Comm;
  Comm cc(MPI_COMM_WORLD);
  BCRSMat mat = setupAnisotropic2d<BCRSMat>(N, indices, cc, &n);
  Vector b(indices.size());

  remoteIndices.rebuild<false>();

  typedef Dune::NegateSet<Communication::OwnerSet> OverlapFlags;
  typedef Dune::OverlappingSchwarzOperator<BCRSMat,Vector,Vector,Communication> Operator;
  typedef Dune::Amg::MatrixHierarchy<Operator,Communication> Hierarchy;
  typedef Dune::Amg::Hierarchy<Vector> VHierarchy;

  Operator op(mat, pinfo);
  Hierarchy hierarchy(op, pinfo);
  VHierarchy vh(b);

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
  MPI_Init(&argc, &argv);

  const int BS=1;
  int N=10;

  if(argc>1)
    N = atoi(argv[1]);

  int procs, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);

  testHierarchy<BS>(N);

  MPI_Finalize();

}
