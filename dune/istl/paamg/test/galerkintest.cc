// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include <config.h>

#include <iostream>
#include <tuple>
#include <dune/common/enumset.hh>
#include <dune/common/propertymap.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/istl/paamg/galerkin.hh>
#include <dune/istl/paamg/dependency.hh>
#include <dune/istl/paamg/globalaggregates.hh>
#include <dune/istl/paamg/pinfo.hh>
#include <dune/istl/io.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/paamg/indicescoarsener.hh>
#include <dune/istl/paamg/galerkin.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include "anisotropic.hh"

template<int BS>
void testCoarsenIndices(int N)
{

  int procs, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);


  typedef Dune::OwnerOverlapCopyCommunication<int> ParallelInformation;
  typedef typename ParallelInformation::ParallelIndexSet ParallelIndexSet;
  typedef typename ParallelInformation::RemoteIndices RemoteIndices;
  typedef Dune::FieldMatrix<double,BS,BS> Block;
  typedef Dune::BCRSMatrix<Block> BCRSMat;
  int n;

  ParallelInformation pinfo(MPI_COMM_WORLD);
  ParallelIndexSet& indices = pinfo.indexSet();
  RemoteIndices& remoteIndices = pinfo.remoteIndices();

  typedef Dune::Communication<MPI_Comm> Comm;
  Comm cc(MPI_COMM_WORLD);

  BCRSMat mat = setupAnisotropic2d<Block>(N, indices, cc, &n);

  pinfo.remoteIndices().template rebuild<false>();

  typedef Dune::Amg::MatrixGraph<BCRSMat> MatrixGraph;
  typedef Dune::Amg::SubGraph<Dune::Amg::MatrixGraph<BCRSMat>,std::vector<bool> > SubGraph;
  typedef Dune::Amg::PropertiesGraph<SubGraph,Dune::Amg::VertexProperties,
      Dune::Amg::EdgeProperties, Dune::IdentityMap, typename SubGraph::EdgeIndexMap> PropertiesGraph;
  typedef typename PropertiesGraph::VertexDescriptor Vertex;
  typedef Dune::Amg::SymmetricCriterion<BCRSMat,Dune::Amg::FirstDiagonal>
  Criterion;

  MatrixGraph mg(mat);
  std::vector<bool> excluded(mat.N());

  typedef ParallelIndexSet::iterator IndexIterator;
  IndexIterator iend = indices.end();
  typename std::vector<bool>::iterator iter=excluded.begin();

  for(IndexIterator index = indices.begin(); index != iend; ++index, ++iter)
    *iter = (index->local().attribute()==GridAttributes::copy);

  SubGraph sg(mg, excluded);
  PropertiesGraph pg(sg, Dune::IdentityMap(), sg.getEdgeIndexMap());
  Dune::Amg::AggregatesMap<Vertex> aggregatesMap(pg.noVertices());

  std::cout << "fine indices: "<<indices << std::endl;
  std::cout << "fine remote: "<<remoteIndices << std::endl;

  int noAggregates, isoAggregates, oneAggregates, skipped;

  std::tie(noAggregates, isoAggregates, oneAggregates,skipped) = aggregatesMap.buildAggregates(mat, pg, Criterion(), false);

  Dune::Amg::printAggregates2d(aggregatesMap, n, N, std::cout);

  ParallelInformation coarseInfo(MPI_COMM_WORLD);
  ParallelIndexSet&      coarseIndices = coarseInfo.indexSet();
  RemoteIndices& coarseRemote = coarseInfo.remoteIndices();

  typename Dune::PropertyMapTypeSelector<Dune::Amg::VertexVisitedTag,PropertiesGraph>::Type visitedMap = Dune::get(Dune::Amg::VertexVisitedTag(), pg);

  pinfo.buildGlobalLookup(aggregatesMap.noVertices());

  int noCoarseVertices = Dune::Amg::IndicesCoarsener<ParallelInformation,Dune::EnumItem<GridFlag,GridAttributes::copy> >::coarsen(pinfo,
                                                                                                                                  pg,
                                                                                                                                  visitedMap,
                                                                                                                                  aggregatesMap,
                                                                                                                                  coarseInfo,
                                                                                                                                  noAggregates);

  coarseInfo.buildGlobalLookup(noCoarseVertices);
  std::cout << rank <<": coarse indices: " <<coarseIndices << std::endl;
  std::cout << rank <<": coarse remote indices:"<<coarseRemote <<std::endl;

  typedef Dune::Interface Interface;
  typedef Dune::BufferedCommunicator Communicator;
  Interface interface;
  interface.build(remoteIndices, Dune::EnumItem<GridFlag,GridAttributes::owner>(), Dune::EnumItem<GridFlag, GridAttributes::copy>());
  Communicator communicator;

  typedef Dune::Amg::GlobalAggregatesMap<Vertex,ParallelIndexSet> GlobalMap;

  GlobalMap gmap(aggregatesMap, coarseInfo.globalLookup());

  communicator.build<GlobalMap>(interface);

  Dune::Amg::printAggregates2d(aggregatesMap, n, N, std::cout);

  communicator.template forward<Dune::Amg::AggregatesGatherScatter<typename MatrixGraph::VertexDescriptor,ParallelIndexSet> >(gmap);

  std::cout<<"Communicated: ";
  Dune::Amg::printAggregates2d(aggregatesMap, n, N, std::cout);

  Dune::Amg::GalerkinProduct<ParallelInformation> productBuilder;

  typedef Dune::IteratorPropertyMap<bool*, Dune::IdentityMap> VisitedMap2;
  /*
     Vector visited;
     visited.reserve(mg.maxVertex());
     Iterator visitedIterator=visited.begin();
   */
  std::cout<<n*n<<"=="<<mg.noVertices()<<std::endl;

  assert(mat.N()==mg.maxVertex()+1);

  bool* visitedIterator = new bool[mat.N()];

  for(Vertex i=0; i < mat.N(); ++i)
    visitedIterator[i]=false;

  VisitedMap2 visitedMap2(visitedIterator, Dune::IdentityMap());

  BCRSMat* coarseMat = productBuilder.build(mg, visitedMap2, pinfo,
                                            aggregatesMap, noCoarseVertices,
                                            Dune::EnumItem<GridFlag,GridAttributes::copy>());

  delete[] visitedIterator;
  pinfo.freeGlobalLookup();
  productBuilder.calculate(mat, aggregatesMap, *coarseMat, coarseInfo, Dune::EnumItem<GridFlag,GridAttributes::copy>());

  if(N<5) {
    Dune::printmatrix(std::cout,mat,"fine","row",9,1);
    Dune::printmatrix(std::cout,*coarseMat,"coarse","row",9,1);
  }
}


int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int N=5;

  if(argc>1)
    N = atoi(argv[1]);
  std::cout << "Galerkin test with N=" << N << std::endl;
  testCoarsenIndices<1>(N);
  MPI_Finalize();
}
