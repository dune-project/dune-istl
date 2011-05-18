// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include "mpi.h"
#include <dune/istl/io.hh>
#include <dune/istl/bvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/paamg/test/anisotropic.hh>
#include <dune/common/timer.hh>
#include <dune/istl/matrixmarket.hh>

#include <iterator>

int main(int argc, char** argv)
{

  MPI_Init(&argc, &argv);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int BS=1;
  int N=100;

  if(argc>1)
    N = atoi(argv[1]);
  std::cout<<"testing for N="<<N<<" BS="<<1<<std::endl;


  typedef Dune::FieldMatrix<double,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<double,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> BVector;
  typedef int GlobalId;
  typedef Dune::OwnerOverlapCopyCommunication<GlobalId> Communication;
  Communication comm(MPI_COMM_WORLD);
  int n;

  std::cout<<comm.communicator().rank()<<" "<<comm.communicator().size()<<
  " "<<size<<std::endl;

  BCRSMat mat = setupAnisotropic2d<BS,double>(N, comm.indexSet(), comm.communicator(), &n, .011);

  BVector bv = BVector(mat.N());
  typedef BVector::iterator VIter;

  int i=0;
  for(VIter entry=bv.begin(); bv.end() != entry; ++entry) {
    typedef BVector::block_type::iterator SIter;
    for(SIter sentry=entry->begin(); sentry != entry->end(); ++sentry,++i)
      *sentry=i;
  }
  comm.remoteIndices().rebuild<false>();
  comm.copyOwnerToAll(bv,bv);

  storeMatrixMarket(mat, std::string("testmat"), comm);
  storeMatrixMarket(bv, std::string("testvec"), comm, false);

  BCRSMat mat1;
  BVector bv1;

  Communication comm1(MPI_COMM_WORLD);

  loadMatrixMarket(mat1, std::string("testmat"), comm1);
  loadMatrixMarket(bv1, std::string("testvec"), comm1, false);

  int ret=0;
  for(VIter entry=bv.begin(), entry1=bv1.begin(); bv.end() != entry; ++entry, ++entry1)
    if(*entry!=*entry1)
    {
      std::cerr<<"written and read vector do not match"<<std::endl;
      ++ret;
    }

  if(comm1.indexSet()!=comm.indexSet())
  {
    std::cerr<<"written and read idxset do not match"<<std::endl;
    ++ret;
  }
  storeMatrixMarket(mat1, std::string("testmat1"), comm1);
  if(ret!=0)
    MPI_Abort(MPI_COMM_WORLD, ret);
  MPI_Finalize();
}
