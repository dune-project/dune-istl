// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include <iostream>
#include <dune/common/mpihelper.hh> // An initializer of MPI
#include <dune/common/exceptions.hh> // We use exceptions
#include <dune/istl/indexset.hh>
#include <dune/istl/remoteindices.hh>
#include <dune/istl/communicator.hh>
#include <dune/istl/plocalindex.hh>
#include <dune/istl/interface.hh>
#include <dune/common/enumset.hh>

enum Flags { owner, ghost };

template<typename T>
struct AddData {
  typedef typename T::value_type IndexedType;

  static double gather(const T& v, int i){
    return v[i];
  }

  static void scatter(T& v, double item, int i){
    v[i]+=item;
  }
};

template<typename T>
struct CopyData {
  typedef typename T::value_type IndexedType;

  static double gather(const T& v, int i){
    return v[i];
  }

  static void scatter(T& v, double item, int i){
    v[i]=item;
  }
};


template<class T>
void doCalculations(T& t){}

void test()
{
  int rank;
  MPI_Comm comm=(MPI_COMM_WORLD);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  using namespace Dune;
  // shortcut for index set type
  typedef ParallelLocalIndex<Flags> LocalIndex;
  typedef ParallelIndexSet<int, LocalIndex > PIndexSet;
  PIndexSet sis;
  sis.beginResize();
  if(rank==0) {

    sis.add(11, LocalIndex(0, ghost));
    for(int i=1; i<=6; i++)
      sis.add(i-1, LocalIndex(i, owner, i<=1||i>5));
    sis.add(6, LocalIndex(7, ghost));
  }else{
    sis.add(5, LocalIndex(0, ghost));
    for(int i=1; i<=6; i++)
      sis.add(5+i, LocalIndex(i, owner, i<=1||i>5));
    sis.add(0,LocalIndex(7, ghost));
  }
  sis.endResize();

  PIndexSet tis;
  tis.beginResize();
  int l=0;
  for(int i=0; i<2; ++i)
    for(int j=0; j<5; ++j) {
      int g=rank*3-1+i*6+j;
      if(g<0||g>11)
        continue;
      Flags flag=(j>0&&j<4) ? owner : ghost;
      tis.add(g, LocalIndex(l++, flag));
    }
  tis.endResize();
  std::cout<<rank<<" isxset: "<<sis<<std::endl;

  RemoteIndices<PIndexSet> riRedist(sis, tis, comm);
  riRedist.rebuild<true>();

  std::vector<int> v;
  RemoteIndices<PIndexSet> riS(sis,sis, comm, v, true);
  riS.rebuild<false>();

  std::cout<<std::endl<<"begin"<<rank<<" riS="<<riS<<" end"<<rank<<std::endl<<std::endl;

  Combine<EnumItem<Flags,ghost>,EnumItem<Flags,owner>,Flags> ghostFlags;
  EnumItem<Flags,owner> ownerFlags;
  Combine<EnumItem<Flags,ghost>, EnumItem<Flags,owner> > allFlags;

  Interface infRedist;
  Interface infS;

  infRedist.build(riRedist, ownerFlags, allFlags);
  infS.build(riS, ownerFlags, ghostFlags);

  std::cout<<"inf "<<rank<<": "<<infS<<std::endl;

  typedef std::vector<double> Container;
  Container s(sis.size(),3), t(tis.size());

  s[sis.size()-1]=-1;

  BufferedCommunicator bComm;
  BufferedCommunicator bCommRedist;
  bComm.build(s, s, infS);
  //bCommRedist.build(s, t, infRedist);
  for(std::size_t i=0; i<sis.size(); i++)
    std::cout<<s[i]<<" ";
  std::cout<<std::endl;

  bComm.forward<CopyData<Container> >(s,s);

  for(std::size_t i=0; i<sis.size(); i++)
    std::cout<<s[i]<<" ";
  std::cout<<std::endl;
  //bCommRedist.forward<CopyData<Container> >(s,t);
  // calculate on the redistributed array
  doCalculations(t);
  //bCommRedist.backward<AddData<Container> >(s,t);
}

int main(int argc, char** argv)
{
  try{
    using namespace Dune;

    //Maybe initialize Mpi
    MPIHelper& helper = MPIHelper::instance(argc, argv);
    std::cout << "Hello World! This is poosc08." << std::endl;
    if(Dune::MPIHelper::isFake)
      std::cout<< "This is a sequential program." << std::endl;
    else{
      test();
    }
    return 0;
  }
  catch (Dune::Exception &e) {
    std::cerr << "Dune reported error: " << e << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown exception thrown!" << std::endl;
  }
}
