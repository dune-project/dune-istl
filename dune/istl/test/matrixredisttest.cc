// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#define DEBUG_REPART

#include <dune/istl/matrixredistribute.hh>
#include <iostream>
#include <dune/istl/paamg/test/anisotropic.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixutils.hh>
#include <dune/istl/paamg/graph.hh>
#include <dune/istl/io.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/bigunsignedint.hh>

class MPIError {
public:
  /** @brief Constructor. */
  MPIError(std::string s, int e) : errorstring(s), errorcode(e){}
  /** @brief The error string. */
  std::string errorstring;
  /** @brief The mpi error code. */
  int errorcode;
};

void MPI_err_handler(MPI_Comm *, int *err_code, ...){
  char *err_string=new char[MPI_MAX_ERROR_STRING];
  int err_length;
  MPI_Error_string(*err_code, err_string, &err_length);
  std::string s(err_string, err_length);
  std::cerr << "An MPI Error occurred:"<<std::endl<<s<<std::endl;
  delete[] err_string;
  throw MPIError(s, *err_code);
}

template<class MatrixBlock>
int testRepart(int N, int coarsenTarget)
{

  std::cout<<"==================================================="<<std::endl;
  std::cout<<" N="<<N<<" coarsenTarget="<<coarsenTarget<<std::endl;

  int procs, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);

  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::bigunsignedint<56> GlobalId;
  typedef Dune::OwnerOverlapCopyCommunication<GlobalId> Communication;
  int n;

  Communication comm(MPI_COMM_WORLD);

  BCRSMat mat = setupAnisotropic2d<MatrixBlock>(N, comm.indexSet(), comm.communicator(), &n, 1);
  typedef typename Dune::Amg::MatrixGraph<BCRSMat> MatrixGraph;

  MatrixGraph graph(mat);
  std::shared_ptr<Communication> coarseComm;

  comm.remoteIndices().template rebuild<false>();

  std::cout<<comm.communicator().rank()<<comm.indexSet()<<std::endl;

  Dune::RedistributeInformation<Communication> ri;
  Dune::graphRepartition(graph, comm, coarsenTarget,
                         coarseComm, ri.getInterface());

  std::cout<<coarseComm->communicator().rank()<<coarseComm->indexSet()<<std::endl;
  BCRSMat newMat;

  if(comm.communicator().rank()==0)
    std::cout<<"Original matrix"<<std::endl;
  comm.communicator().barrier();
  printGlobalSparseMatrix(mat, comm, std::cout);


  redistributeMatrix(mat, newMat, comm, *coarseComm, ri);

  std::cout<<comm.communicator().rank()<<": redist interface "<<ri.getInterface()<<std::endl;

  if(comm.communicator().rank()==0)
    std::cout<<"Redistributed matrix"<<std::endl;
  comm.communicator().barrier();
  if(coarseComm->communicator().size()>0)
    printGlobalSparseMatrix(newMat, *coarseComm, std::cout);
  comm.communicator().barrier();

  // Check for symmetry
  int ret=0;
  typedef typename BCRSMat::ConstRowIterator RIter;
  for(RIter row=newMat.begin(), rend=newMat.end(); row != rend; ++row) {
    typedef typename BCRSMat::ConstColIterator CIter;
    for(CIter col=row->begin(), cend=row->end(); col!=cend; ++col)
    {
      if(col.index()<=row.index())
        try{
          newMat[col.index()][row.index()];
        }catch(const Dune::ISTLError&) {
          std::cerr<<coarseComm->communicator().rank()<<": entry ("
                   <<col.index()<<","<<row.index()<<") missing!"<<std::endl;
          ret=1;

        }
      else
        break;
    }
  }

  //if(coarseComm->communicator().rank()==0)
  //Dune::printmatrix(std::cout, newMat, "redist", "row");
  return ret;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Errhandler handler;
  MPI_Comm_create_errhandler(MPI_err_handler, &handler);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, handler);
  int procs;
  MPI_Comm_size(MPI_COMM_WORLD, &procs);

  int N=4*procs;

  int coarsenTarget=1;

  if(argc>1)
    N = atoi(argv[1]);
  if(argc>2)
    coarsenTarget = atoi(argv[2]);

  if(N<procs*2) {
    std::cerr<<"Problem size insufficient for process number"<<std::endl;
    return 1;
  }

  testRepart<Dune::FieldMatrix<double, 1, 1>>(N,coarsenTarget);
  testRepart<Dune::FieldMatrix<double, 2, 2>>(N/2,coarsenTarget);
  testRepart<double>(N,coarsenTarget);
  MPI_Finalize();
}
