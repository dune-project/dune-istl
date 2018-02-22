// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/bvector.hh>
#include <dune/istl/superlu.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/paamg/graph.hh>
#include <dune/istl/matrixredistribute.hh>
#include <dune/istl/asyncblockpreconditioner.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/asyncbufferexchange.hh>
#include <dune/istl/asyncloopsolver.hh>
#include <dune/istl/asyncstoppingcriteria.hh>
#include "laplacian.hh"

template<class C, class M>
auto redistributeMatrix(C& comm, M& in, M& out){
  // redistribute
  typedef std::size_t GlobalId; // The type for the global index
  typedef Dune::OwnerOverlapCopyCommunication<GlobalId> OOC;
  OOC ooc(comm);
  OOC* ooc_redist;

  typedef Dune::Amg::MatrixGraph<M> MatrixGraph;
  Dune::RedistributeInformation<OOC> rinfo;

  bool hasDofs = Dune::graphRepartition(MatrixGraph(in), ooc,
                                        static_cast<int>(comm.size()),
                                        ooc_redist,
                                        rinfo.getInterface(),
                                        false);
  rinfo.setSetup();
  redistributeMatrix(in, out, ooc, *ooc_redist, rinfo);
  std::cout << "Rank " << comm.rank() << ": " << out.N() <<
    " x "<< out.M() << " #NNZ:" << out.nonzeroes() << std::endl;
  return ooc_redist;
}

int main(int argc, char** argv)
{
  Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);
  auto world = helper.getCommunicator();
  int rank = world.rank();
  int size = world.size();

  constexpr int BS=1;
  int N=40;

  if(argc>1)
    N = atoi(argv[1]);
  std::cout<<"testing for N="<<N<<" BS="<<1<<std::endl;

  typedef Dune::FieldMatrix<double,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<double,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> BVector;

  BCRSMat mat, mat_dist;
  if(world.rank() == 0)
    setupLaplacian(mat,N);
  // distribute
  auto ooc = redistributeMatrix(world, mat, mat_dist);
  N = mat_dist.M();

  typedef typename std::remove_reference_t<decltype(*ooc)>::PIS PIS;
  typedef Dune::AsyncBufferedExchange<BVector, decltype(world), PIS> EX;
  typedef Dune::OverlappingSchwarzOperator<BCRSMat,BVector,BVector, EX> Operator;

  PIS pis = ooc->indexSet();
  Dune::RemoteIndices<PIS> ri(pis, pis, world);
  ri.template rebuild<false>();
  std::shared_ptr<Dune::Interface> pInterface = std::make_shared<Dune::Interface>(world);
  Dune::EnumItem<Dune::OwnerOverlapCopyAttributeSet::AttributeSet,Dune::OwnerOverlapCopyAttributeSet::owner> ownerFlags;
  Dune::AllSet<Dune::OwnerOverlapCopyAttributeSet::AttributeSet> allFlags;
  pInterface->build(ri, ownerFlags, allFlags);
  EX ex(world, 4712, pInterface, pis);

  Operator fop(mat_dist, ex);
  BVector b(N), x(N);
  b=1;
  x=0;

  Dune::InverseOperatorResult res;
  typedef Dune::SeqGS<BCRSMat,BVector,BVector> SEQ_PREC;
  SEQ_PREC seq_prec(mat_dist, 1,1.0);
  Dune::AsyncBlockPreconditioner<BVector, BVector, SEQ_PREC> prec0(seq_prec, ex);
  Dune::AsyncNormCheck<BVector, decltype(world)> anc(world, 1e-7);
  Dune::AsyncLoopSolver<BVector> solver0(fop, prec0, anc, 5000, 3);

  solver0.apply(x, b, res);

  // check result
  fop.apply(x, b);
  x = 1;
  b -= x;
  ex.project(b);
  Dune::CollectiveCommunication<decltype(world)> cc;
  auto norm = std::sqrt(cc.sum(b.two_norm2()));

  std::cout << "|b|" << norm << std::endl;

  return 0;
}
