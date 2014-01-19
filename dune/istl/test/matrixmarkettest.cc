// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include <iterator>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/float_cmp.hh>

#include <dune/istl/matrixmarket.hh>
#include <dune/istl/io.hh>
#include <dune/istl/bvector.hh>

#if HAVE_MPI
#include <dune/istl/paamg/test/anisotropic.hh>
#include "mpi.h"
#include <dune/istl/schwarz.hh>
#else
#include <dune/istl/operators.hh>
#include "laplacian.hh"
#endif

int main(int argc, char** argv)
{
#if HAVE_MPI
  MPI_Init(&argc, &argv);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
  const int BS=1;
  int N=100;

  if(argc>1)
    N = atoi(argv[1]);
  std::cout<<"testing for N="<<N<<" BS="<<1<<std::endl;


  typedef Dune::FieldMatrix<double,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<double,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> BVector;

#if HAVE_MPI
  typedef int GlobalId;
  typedef Dune::OwnerOverlapCopyCommunication<GlobalId> Communication;
  Communication comm(MPI_COMM_WORLD);
  std::cout<<comm.communicator().rank()<<" "<<comm.communicator().size()<<
  " "<<size<<std::endl;
  int n;
  BCRSMat mat = setupAnisotropic2d<BCRSMat>(N, comm.indexSet(), comm.communicator(), &n, .011);
#else
  BCRSMat mat;
  setupLaplacian(mat, N);
#endif

  BVector bv(mat.N()), cv(mat.N());
  typedef BVector::iterator VIter;

  int i=0;
  for(VIter entry=bv.begin(); bv.end() != entry; ++entry) {
    typedef BVector::block_type::iterator SIter;
    for(SIter sentry=entry->begin(); sentry != entry->end(); ++sentry,++i)
      *sentry=i;
  }

#if HAVE_MPI
  comm.remoteIndices().rebuild<false>();
  comm.copyOwnerToAll(bv,bv);

  Dune::OverlappingSchwarzOperator<BCRSMat,BVector,BVector,Communication> op(mat, comm);
  op.apply(bv, cv);
  storeMatrixMarket(mat, std::string("testmat"), comm);
  storeMatrixMarket(bv, std::string("testvec"), comm, false);
#else
  typedef Dune::MatrixAdapter<BCRSMat,BVector,BVector> Operator;
  Operator op(mat);
  op.apply(bv, cv);

  storeMatrixMarket(mat, std::string("testmat"));
  storeMatrixMarket(bv, std::string("testvec"));
#endif

  BCRSMat mat1;
  BVector bv1,cv1;

#if HAVE_MPI
  Communication comm1(MPI_COMM_WORLD);

  loadMatrixMarket(mat1, std::string("testmat"), comm1);
  loadMatrixMarket(bv1, std::string("testvec"), comm1, false);
#else
  loadMatrixMarket(mat1, std::string("testmat"));
  loadMatrixMarket(bv1, std::string("testvec"));
#endif

  int ret=0;
  if(mat.N()!=mat1.N() || mat.M()!=mat1.M())
  {
    ++ret;
    std::cerr<<"matrix sizes do not match"<<std::endl;
  }
  typedef BCRSMat::const_iterator RowIterator;
  typedef BCRSMat::ConstColIterator ColIterator;

  for(RowIterator row=mat.begin(), row1=mat1.begin(); row!=mat.end(); ++row, ++row1)
    for(ColIterator col=row->begin(), col1=row1->begin(); col!= row->end(); ++col, ++col1)
    {
      if(col.index()!=col1.index()) {
        std::cerr <<"Column indices do not match"<<std::endl;
        ++ret;
      }
      if(!Dune::FloatCmp::eq(*col, *col1)) {
        std::cerr <<"Matrix entries do not match"<<std::endl;
        ++ret;
      }
    }

  for(VIter entry=bv.begin(), entry1=bv1.begin(); bv.end() != entry; ++entry, ++entry1)
    if(*entry!=*entry1)
    {
      std::cerr<<"written and read vector do not match"<<std::endl;
      ++ret;
    }

  cv1.resize(mat1.M());

#if HAVE_MPI
  Dune::OverlappingSchwarzOperator<BCRSMat,BVector,BVector,Communication> op1(mat1, comm1);
  op1.apply(bv1, cv1);

  if(comm1.indexSet()!=comm.indexSet())
  {
    std::cerr<<"written and read idxset do not match"<<std::endl;
    ++ret;
  }
#else
  typedef Dune::MatrixAdapter<BCRSMat,BVector,BVector> Operator;
  Operator op1(mat1);
  op1.apply(bv1, cv1);
#endif

  for(VIter entry=cv.begin(), entry1=cv1.begin(); cv.end() != entry; ++entry, ++entry1)
    if(*entry!=*entry1)
    {
      std::cerr<<"computed vectors do not match"<<std::endl;
      ++ret;
    }

#if HAVE_MPI
  if(ret!=0)
    MPI_Abort(MPI_COMM_WORLD, ret);
  MPI_Finalize();
#endif
}
