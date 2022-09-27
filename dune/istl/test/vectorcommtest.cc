// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/remoteindices.hh>
#include <vector>
#include <dune/common/enumset.hh>
#include <dune/common/fvector.hh>
#include <algorithm>
#include <iostream>
#include "mpi.h"

enum GridFlags {
  owner, overlap, border
};

void testIndices(MPI_Comm comm)
{
  //using namespace Dune;

  // The global grid size
  const int Nx = 20;
  const int Ny = 2;

  // Process configuration
  int procs, rank, master=0;
  MPI_Comm_size(comm, &procs);
  MPI_Comm_rank(comm, &rank);

  // shift the ranks
  //rank = (rank + 1) % procs;
  //master= (master+1) %procs;

  // The local grid
  int nx = Nx/procs;
  // distributed indexset
  //  typedef ParallelLocalIndex<GridFlags> LocalIndexType;

  typedef Dune::ParallelIndexSet<int,Dune::ParallelLocalIndex<GridFlags>,45> ParallelIndexSet;

  ParallelIndexSet distIndexSet;
  // global indexset
  ParallelIndexSet globalIndexSet;

  // Set up the indexsets.
  int start = std::max(rank*nx-1,0);
  int end = std::min((rank + 1) * nx+1, Nx);

  distIndexSet.beginResize();

  int localIndex=0;
  int size = Ny*(end-start);

  typedef Dune::FieldVector<int,5> Vector;
  typedef std::vector<Vector> Array;

  Array distArray(size);
  Array* globalArray;
  int index=0;

  for(int j=0; j<Ny; j++)
    for(int i=start; i<end; i++) {
      bool isPublic = (i<=start+1)||(i>=end-2);
      GridFlags flag = owner;
      if((i==start && i!=0)||(i==end-1 && i!=Nx-1)) {
        distArray[index++]=-(i+j*Nx+rank*Nx*Ny);
        flag = overlap;
      }else
        distArray[index++]=i+j*Nx+rank*Nx*Ny;

      distIndexSet.add(i+j*Nx, Dune::ParallelLocalIndex<GridFlags> (localIndex++,flag,isPublic));
    }

  distIndexSet.endResize();

  if(rank==master) {
    // build global indexset on first process
    globalIndexSet.beginResize();
    globalArray=new Array(Nx*Ny);
    int k=0;
    for(int j=0; j<Ny; j++)
      for(int i=0; i<Nx; i++) {
        globalIndexSet.add(i+j*Nx, Dune::ParallelLocalIndex<GridFlags> (i+j*Nx,owner,false));
        globalArray->operator[](i+j*Nx)=-(i+j*Nx);
        k++;

      }

    globalIndexSet.endResize();
  }else
    globalArray=new Array(1); // Size one is needed for CommPolicy

  typedef Dune::RemoteIndices<ParallelIndexSet> RemoteIndices;

  RemoteIndices accuIndices(distIndexSet, globalIndexSet,  comm);
  RemoteIndices overlapIndices(distIndexSet, distIndexSet, comm);
  accuIndices.rebuild<true>();
  overlapIndices.rebuild<false>();

  Dune::DatatypeCommunicator<ParallelIndexSet> accumulator, overlapExchanger;

  Dune::EnumItem<GridFlags,owner> sourceFlags;
  Dune::Combine<Dune::EnumItem<GridFlags,overlap>,Dune::EnumItem<GridFlags,owner>,GridFlags> destFlags;

  accumulator.build(accuIndices, sourceFlags, distArray, destFlags, *globalArray);

  overlapExchanger.build(overlapIndices, Dune::EnumItem<GridFlags,owner>(), distArray, Dune::EnumItem<GridFlags,overlap>(), distArray);

  std::cout<< rank<<": before forward distArray=";
  std::copy(distArray.begin(), distArray.end(), std::ostream_iterator<Vector>(std::cout, " "));

  // Exchange the overlap
  overlapExchanger.forward();

  std::cout<<rank<<": overlap exchanged distArray=";
  std::copy(distArray.begin(), distArray.end(), std::ostream_iterator<Vector>(std::cout, " "));

  if(rank==master)
  {
    std::cout<<": before forward globalArray=";
    std::copy(globalArray->begin(), globalArray->end(), std::ostream_iterator<Vector>(std::cout, " "));
  }

  accumulator.forward();


  if(rank==master) {
    std::cout<<"after forward global: ";
    std::copy(globalArray->begin(), globalArray->end(), std::ostream_iterator<Vector>(std::cout, " "));
    struct Twice
    {
      void operator()(Vector& v)
      {
        v*=2;
      }
    };

    std::for_each(globalArray->begin(), globalArray->end(), Twice());
    std::cout<<" Multiplied by two: globalArray=";
    std::copy(globalArray->begin(), globalArray->end(), std::ostream_iterator<Vector>(std::cout, " "));
  }

  accumulator.backward();
  std::cout<< rank<<": after backward distArray=";
  std::copy(distArray.begin(), distArray.end(), std::ostream_iterator<Vector>(std::cout, " "));


  // Exchange the overlap
  overlapExchanger.forward();

  std::cout<<rank<<": overlap exchanged distArray=";
  std::copy(distArray.begin(), distArray.end(), std::ostream_iterator<Vector>(std::cout, " "));

  //std::cout << rank<<": source and dest are the same:"<<std::endl;
  //std::cout << remote<<std::endl<<std::flush;
  if(rank==master)
    delete globalArray;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  testIndices(MPI_COMM_WORLD);
  MPI_Finalize();

}
