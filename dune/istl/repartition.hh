// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_REPARTITION_HH
#define DUNE_ISTL_REPARTITION_HH

#include <cassert>
#include <map>
#include <utility>
#include <cmath>

#if HAVE_PARMETIS
// Explicitly use C linkage as scotch does not extern "C" in its headers.
// Works because ParMETIS/METIS checks whether compiler is C++ and otherwise
// does not use extern "C". Therfore no nested extern "C" will be created
extern "C"
{
#include <parmetis.h>
}
#endif

#include <dune/common/timer.hh>
#include <dune/common/enumset.hh>
#include <dune/common/stdstreams.hh>
#include <dune/common/parallel/mpitraits.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/indicessyncer.hh>
#include <dune/common/parallel/remoteindices.hh>
#include <dune/common/rangeutilities.hh>

#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/paamg/graph.hh>

/**
 * @file
 * @brief Functionality for redistributing a parallel index set using graph partitioning.
 *
 * Refactored version of an intern.
 * @author Markus Blatt
 */

namespace Dune
{
  namespace Metis
  {
    // Explicitly specify a real_t and idx_t for older (Par)METIS versions that do not
    // provide these typedefs
#if HAVE_PARMETIS && defined(REALTYPEWIDTH)
    using real_t = ::real_t;
#else
    using real_t = float;
#endif

#if HAVE_PARMETIS && defined(IDXTYPEWIDTH)
    using idx_t = ::idx_t;
#elif HAVE_PARMETIS && defined(HAVE_SCOTCH_NUM_TYPE)
    using idx_t = SCOTCH_Num;
#elif HAVE_PARMETIS
    using idx_t = int;
#else
    using idx_t = std::size_t;
#endif
  }


#if HAVE_MPI
  /**
   * @brief Fills the holes in an index set.
   *
   * In general the index set only needs to know those indices
   * where communication my occur. In usual FE computations these
   * are just those near the processor boundaries.
   *
   * For the repartitioning we need to know all all indices for which data is stored.
   * The missing indices will be created in this method.
   *
   * @param graph The graph to reparition.
   * @param oocomm The communication information.
   */
  template<class G, class T1, class T2>
  void fillIndexSetHoles(const G& graph, Dune::OwnerOverlapCopyCommunication<T1,T2>& oocomm)
  {
    typedef typename Dune::OwnerOverlapCopyCommunication<T1,T2>::ParallelIndexSet IndexSet;
    typedef typename IndexSet::LocalIndex::Attribute Attribute;

    IndexSet& indexSet = oocomm.indexSet();
    const typename Dune::OwnerOverlapCopyCommunication<T1,T2>::GlobalLookupIndexSet& lookup =oocomm.globalLookup();

    std::size_t sum=0, needed = graph.noVertices()-indexSet.size();
    std::vector<std::size_t> neededall(oocomm.communicator().size(), 0);

    MPI_Allgather(&needed, 1, MPITraits<std::size_t>::getType() , &(neededall[0]), 1, MPITraits<std::size_t>::getType(), oocomm.communicator());
    for(int i=0; i<oocomm.communicator().size(); ++i)
      sum=sum+neededall[i];   // MAke this for generic

    if(sum==0)
      // Nothing to do
      return;

    //Compute Maximum Global Index
    T1 maxgi=0;
    auto end = indexSet.end();
    for(auto it = indexSet.begin(); it != end; ++it)
      maxgi=std::max(maxgi,it->global());

    //Process p creates global indices consecutively
    //starting atmaxgi+\sum_{i=1}^p neededall[i]
    // All created indices are owned by the process
    maxgi=oocomm.communicator().max(maxgi);
    ++maxgi;  //Sart with the next free index.

    for(int i=0; i<oocomm.communicator().rank(); ++i)
      maxgi=maxgi+neededall[i];   // TODO: make this more generic

    // Store the global index information for repairing the remote index information
    std::map<int,SLList<std::pair<T1,Attribute> > > globalIndices;
    storeGlobalIndicesOfRemoteIndices(globalIndices, oocomm.remoteIndices());
    indexSet.beginResize();

    for(auto vertex = graph.begin(), vend=graph.end(); vertex != vend; ++vertex) {
      const typename IndexSet::IndexPair* pair=lookup.pair(*vertex);
      if(pair==0) {
        // No index yet, add new one
        indexSet.add(maxgi, typename IndexSet::LocalIndex(*vertex, OwnerOverlapCopyAttributeSet::owner, false));
        ++maxgi;
      }
    }

    indexSet.endResize();

    repairLocalIndexPointers(globalIndices, oocomm.remoteIndices(), indexSet);

    oocomm.freeGlobalLookup();
    oocomm.buildGlobalLookup();
#ifdef DEBUG_REPART
    std::cout<<"Holes are filled!"<<std::endl;
    std::cout<<oocomm.communicator().rank()<<": "<<oocomm.indexSet()<<std::endl;
#endif
  }

  namespace
  {

    class ParmetisDuneIndexMap
    {
    public:
      template<class Graph, class OOComm>
      ParmetisDuneIndexMap(const Graph& graph, const OOComm& com);
      int toParmetis(int i) const
      {
        return duneToParmetis[i];
      }
      int toLocalParmetis(int i) const
      {
        return duneToParmetis[i]-base_;
      }
      int operator[](int i) const
      {
        return duneToParmetis[i];
      }
      int toDune(int i) const
      {
        return parmetisToDune[i];
      }
      std::vector<int>::size_type numOfOwnVtx() const
      {
        return parmetisToDune.size();
      }
      Metis::idx_t* vtxDist()
      {
        return &vtxDist_[0];
      }
      int globalOwnerVertices;
    private:
      int base_;
      std::vector<int> duneToParmetis;
      std::vector<int> parmetisToDune;
      // range of vertices for processor i: vtxdist[i] to vtxdist[i+1] (parmetis global)
      std::vector<Metis::idx_t> vtxDist_;
    };

    template<class G, class OOComm>
    ParmetisDuneIndexMap::ParmetisDuneIndexMap(const G& graph, const OOComm& oocomm)
      : duneToParmetis(graph.noVertices(), -1), vtxDist_(oocomm.communicator().size()+1)
    {
      int npes=oocomm.communicator().size(), mype=oocomm.communicator().rank();

      typedef typename OOComm::OwnerSet OwnerSet;

      int numOfOwnVtx=0;
      auto end = oocomm.indexSet().end();
      for(auto index = oocomm.indexSet().begin(); index != end; ++index) {
        if (OwnerSet::contains(index->local().attribute())) {
          numOfOwnVtx++;
        }
      }
      parmetisToDune.resize(numOfOwnVtx);
      std::vector<int> globalNumOfVtx(npes);
      // make this number available to all processes
      MPI_Allgather(&numOfOwnVtx, 1, MPI_INT, &(globalNumOfVtx[0]), 1, MPI_INT, oocomm.communicator());

      int base=0;
      vtxDist_[0] = 0;
      for(int i=0; i<npes; i++) {
        if (i<mype) {
          base += globalNumOfVtx[i];
        }
        vtxDist_[i+1] = vtxDist_[i] + globalNumOfVtx[i];
      }
      globalOwnerVertices=vtxDist_[npes];
      base_=base;

#ifdef DEBUG_REPART
      std::cout << oocomm.communicator().rank()<<" vtxDist: ";
      for(int i=0; i<= npes; ++i)
        std::cout << vtxDist_[i]<<" ";
      std::cout<<std::endl;
#endif

      // Traverse the graph and assign a new consecutive number/index
      // starting by "base" to all owner vertices.
      // The new index is used as the ParMETIS global index and is
      // stored in the vector "duneToParmetis"
      auto vend = graph.end();
      for(auto vertex = graph.begin(); vertex != vend; ++vertex) {
        const typename OOComm::ParallelIndexSet::IndexPair* index=oocomm.globalLookup().pair(*vertex);
        assert(index);
        if (OwnerSet::contains(index->local().attribute())) {
          // assign and count the index
          parmetisToDune[base-base_]=index->local();
          duneToParmetis[index->local()] = base++;
        }
      }

      // At this point, every process knows the ParMETIS global index
      // of it's owner vertices. The next step is to get the
      // ParMETIS global index of the overlap vertices from the
      // associated processes. To do this, the Dune::Interface class
      // is used.
#ifdef DEBUG_REPART
      std::cout <<oocomm.communicator().rank()<<": before ";
      for(std::size_t i=0; i<duneToParmetis.size(); ++i)
        std::cout<<duneToParmetis[i]<<" ";
      std::cout<<std::endl;
#endif
      oocomm.copyOwnerToAll(duneToParmetis,duneToParmetis);
#ifdef DEBUG_REPART
      std::cout <<oocomm.communicator().rank()<<": after ";
      for(std::size_t i=0; i<duneToParmetis.size(); ++i)
        std::cout<<duneToParmetis[i]<<" ";
      std::cout<<std::endl;
#endif
    }
  }

  struct RedistributeInterface
    : public Interface
  {
    void setCommunicator(MPI_Comm comm)
    {
      communicator_=comm;
    }
    template<class Flags,class IS>
    void buildSendInterface(const std::vector<int>& toPart, const IS& idxset)
    {
      std::map<int,int> sizes;

      for(auto i=idxset.begin(), end=idxset.end(); i!=end; ++i)
        if(Flags::contains(i->local().attribute()))
          ++sizes[toPart[i->local()]];

      // Allocate the necessary space
      for(auto i=sizes.begin(), end=sizes.end(); i!=end; ++i)
        interfaces()[i->first].first.reserve(i->second);

      //Insert the interface information
      for(auto i=idxset.begin(), end=idxset.end(); i!=end; ++i)
        if(Flags::contains(i->local().attribute()))
          interfaces()[toPart[i->local()]].first.add(i->local());
    }

    void reserveSpaceForReceiveInterface(int proc, int size)
    {
      interfaces()[proc].second.reserve(size);
    }
    void addReceiveIndex(int proc, std::size_t idx)
    {
      interfaces()[proc].second.add(idx);
    }
    template<typename TG>
    void buildReceiveInterface(std::vector<std::pair<TG,int> >& indices)
    {
      std::size_t i=0;
      for(auto idx=indices.begin(); idx!= indices.end(); ++idx) {
        interfaces()[idx->second].second.add(i++);
      }
    }

    ~RedistributeInterface()
    {}

  };

  namespace
  {
    /**
     * @brief Fills send buffer with global indices.
     *
     * @param ownerVec the owner vertices to send
     * @param overlapSet the overlap vertices to send
     * @param sendBuf the send buffer
     * @param buffersize The size of the send buffer
     * @param comm Communicator for the send.
     */
    template<class GI>
    void createSendBuf(std::vector<GI>& ownerVec, std::set<GI>& overlapVec, std::set<int>& neighbors, char *sendBuf, int buffersize, MPI_Comm comm) {
      // Pack owner vertices
      std::size_t s=ownerVec.size();
      int pos=0;
      if(s==0)
        ownerVec.resize(1); // otherwise would read beyond the memory bound
      MPI_Pack(&s, 1, MPITraits<std::size_t>::getType(), sendBuf, buffersize, &pos, comm);
      MPI_Pack(&(ownerVec[0]), s, MPITraits<GI>::getType(), sendBuf, buffersize, &pos, comm);
      s = overlapVec.size();
      MPI_Pack(&s, 1, MPITraits<std::size_t>::getType(), sendBuf, buffersize, &pos, comm);
      for(auto i=overlapVec.begin(), end= overlapVec.end(); i != end; ++i)
        MPI_Pack(const_cast<GI*>(&(*i)), 1, MPITraits<GI>::getType(), sendBuf, buffersize, &pos, comm);

      s=neighbors.size();
      MPI_Pack(&s, 1, MPITraits<std::size_t>::getType(), sendBuf, buffersize, &pos, comm);

      for(auto i=neighbors.begin(), end= neighbors.end(); i != end; ++i)
        MPI_Pack(const_cast<int*>(&(*i)), 1, MPI_INT, sendBuf, buffersize, &pos, comm);
    }
    /**
     * @brief save the values of the received MPI buffer to the owner/overlap vectors
     *
     * @param recvBuf the receive buffer.
     * @param ownerVec the vector to store the owner indices in.
     * @param overlapVec the set to store the overlap indices in.
     * @param comm The communicator used in the receive.
     */
    template<class GI>
    void saveRecvBuf(char *recvBuf, int bufferSize, std::vector<std::pair<GI,int> >& ownerVec,
                     std::set<GI>& overlapVec, std::set<int>& neighbors, RedistributeInterface& inf, int from, MPI_Comm comm) {
      std::size_t size;
      int pos=0;
      // unpack owner vertices
      MPI_Unpack(recvBuf, bufferSize, &pos, &size, 1, MPITraits<std::size_t>::getType(), comm);
      inf.reserveSpaceForReceiveInterface(from, size);
      ownerVec.reserve(ownerVec.size()+size);
      for(; size!=0; --size) {
        GI gi;
        MPI_Unpack(recvBuf, bufferSize, &pos, &gi, 1, MPITraits<GI>::getType(), comm);
        ownerVec.push_back(std::make_pair(gi,from));
      }
      // unpack overlap vertices
      MPI_Unpack(recvBuf, bufferSize, &pos, &size, 1, MPITraits<std::size_t>::getType(), comm);
      typename std::set<GI>::iterator ipos = overlapVec.begin();
      Dune::dverb << "unpacking "<<size<<" overlap"<<std::endl;
      for(; size!=0; --size) {
        GI gi;
        MPI_Unpack(recvBuf, bufferSize, &pos, &gi, 1, MPITraits<GI>::getType(), comm);
        ipos=overlapVec.insert(ipos, gi);
      }
      //unpack neighbors
      MPI_Unpack(recvBuf, bufferSize, &pos, &size, 1,  MPITraits<std::size_t>::getType(), comm);
      Dune::dverb << "unpacking "<<size<<" neighbors"<<std::endl;
      typename std::set<int>::iterator npos = neighbors.begin();
      for(; size!=0; --size) {
        int n;
        MPI_Unpack(recvBuf, bufferSize, &pos, &n, 1, MPI_INT, comm);
        npos=neighbors.insert(npos, n);
      }
    }

    /**
     * @brief Find the optimal domain number for a given process
     *
     * The estimation is necessary because the result of ParMETIS for
     * the new partition is only a domain/set number and not a process number.
     *
     * @param comm the MPI communicator
     * @param *part the result array of the ParMETIS repartition
     * @param numOfOwnVtx the number of owner vertices
     * @param nparts the number of target partitions/processes
     * @param *myDomain the optimal output domain number
     * @param domainMapping[] the array of output domain mapping
     */
    template<typename T>
    void getDomain(const MPI_Comm& comm, T *part, int numOfOwnVtx, int nparts, int *myDomain, std::vector<int> &domainMapping) {
      int npes, mype;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &mype);
      MPI_Status status;

      *myDomain = -1;
      int i=0;
      int j=0;

      std::vector<int> domain(nparts, 0);
      std::vector<int> assigned(npes, 0);
      // init domain Mapping
      domainMapping.assign(domainMapping.size(), -1);

      // count the occurrence of domains
      for (i=0; i<numOfOwnVtx; i++) {
        domain[part[i]]++;
      }

      std::vector<int> domainMatrix(npes * nparts, -1);

      // init buffer with the own domain
      int *buf = new int[nparts];
      for (i=0; i<nparts; i++) {
        buf[i] = domain[i];
        domainMatrix[mype*nparts+i] = domain[i];
      }
      int pe=0;
      int src = (mype-1+npes)%npes;
      int dest = (mype+1)%npes;
      // ring communication, we need n-1 communications for n processors
      for (i=0; i<npes-1; i++) {
        MPI_Sendrecv_replace(buf, nparts, MPI_INT, dest, 0, src, 0, comm, &status);
        // pe is the process of the actual received buffer
        pe = ((mype-1-i)+npes)%npes;
        for(j=0; j<nparts; j++) {
          // save the values to the domain matrix
          domainMatrix[pe*nparts+j] = buf[j];
        }
      }
      delete[] buf;

      // Start the domain calculation.
      // The process which contains the maximum number of vertices of a
      // particular domain is selected to choose it's favorate domain
      int maxOccurance = 0;
      pe = -1;
      std::set<std::size_t> unassigned;

      for(i=0; i<nparts; i++) {
        for(j=0; j<npes; j++) {
          // process has no domain assigned
          if (assigned[j]==0) {
            if (maxOccurance < domainMatrix[j*nparts+i]) {
              maxOccurance = domainMatrix[j*nparts+i];
              pe = j;
            }
          }

        }
        if (pe!=-1) {
          // process got a domain, ...
          domainMapping[i] = pe;
          // ...mark as assigned
          assigned[pe] = 1;
          if (pe==mype) {
            *myDomain = i;
          }
          pe = -1;
        }
        else
        {
          unassigned.insert(i);
        }
        maxOccurance = 0;
      }

      typename std::vector<int>::iterator next_free = assigned.begin();

      for(auto udomain = unassigned.begin(),
            end = unassigned.end(); udomain != end; ++udomain)
      {
        next_free = std::find_if(next_free, assigned.end(), std::bind(std::less<int>(), std::placeholders::_1, 1));
        assert(next_free !=  assigned.end());
        domainMapping[*udomain] = next_free-assigned.begin();
        *next_free = 1;
      }
    }

    struct SortFirst
    {
      template<class T>
      bool operator()(const T& t1, const T& t2) const
      {
        return t1<t2;
      }
    };


    /**
     * @brief Merge the owner/overlap vectors
     *
     * This function merges and adds the vertices of a owner/overlap
     * vector to a result owner/overlap vector
     *
     * @param &ownerVec a global index vector contains the owner vertices to merge/add, sorted according
     * to the global index.
     * @param &overlapSet a global index set contains the overlap vertices to merge/add
     */
    template<class GI>
    void mergeVec(std::vector<std::pair<GI, int> >& ownerVec, std::set<GI>& overlapSet) {

#ifdef DEBUG_REPART
      // Safety check for duplicates.
      if(ownerVec.size()>0)
      {
        auto old=ownerVec.begin();
        for(auto i=old+1, end=ownerVec.end(); i != end; old=i++)
        {
          if(i->first==old->first)
          {
            std::cerr<<"Value at indes"<<old-ownerVec.begin()<<" is the same as at index "
                     <<i-ownerVec.begin()<<" ["<<old->first<<","<<old->second<<"]==["
                     <<i->first<<","<<i->second<<"]"<<std::endl;
            throw "Huch!";
          }
        }
      }

#endif

      auto v=ownerVec.begin(), vend=ownerVec.end();
      for(auto s=overlapSet.begin(), send=overlapSet.end(); s!=send;)
      {
        while(v!=vend && v->first<*s) ++v;
        if(v!=vend && v->first==*s) {
          // Move to the next element before erasing
          // thus s stays valid!
          auto tmp=s;
          ++s;
          overlapSet.erase(tmp);
        }else
          ++s;
      }
    }


    /**
     * @brief get the non-owner neighbors of a given vertex
     *
     * For a given vertex, get the index of all non-owner neighbor vertices are
     * computed.
     *
     * @param g the local graph
     * @param part Where the vertices become owner
     * @param vtx the given vertex
     * @param parmetisVtxMapping mapping between Dune and ParMETIS vertices
     * @param indexSet the indexSet
     * @param neighbor the output set to store the neighbor indices in.
     */
    template<class OwnerSet, class Graph, class IS, class GI>
    void getNeighbor(const Graph& g, std::vector<int>& part,
                     typename Graph::VertexDescriptor vtx, const IS& indexSet,
                     int toPe, std::set<GI>& neighbor, std::set<int>& neighborProcs) {
      for(auto edge=g.beginEdges(vtx), end=g.endEdges(vtx); edge!=end; ++edge)
      {
        const typename IS::IndexPair* pindex = indexSet.pair(edge.target());
        assert(pindex);
        if(part[pindex->local()]!=toPe || !OwnerSet::contains(pindex->local().attribute()))
        {
          // is sent to another process and therefore becomes overlap
          neighbor.insert(pindex->global());
          neighborProcs.insert(part[pindex->local()]);
        }
      }
    }

    template<class T, class I>
    void my_push_back(std::vector<T>& ownerVec, const I& index, [[maybe_unused]] int proc)
    {
      ownerVec.push_back(index);
    }

    template<class T, class I>
    void my_push_back(std::vector<std::pair<T,int> >& ownerVec, const I& index, int proc)
    {
      ownerVec.push_back(std::make_pair(index,proc));
    }
    template<class T>
    void reserve(std::vector<T>&, RedistributeInterface&, int)
    {}
    template<class T>
    void reserve(std::vector<std::pair<T,int> >& ownerVec, RedistributeInterface& redist, int proc)
    {
      redist.reserveSpaceForReceiveInterface(proc, ownerVec.size());
    }


    /**
     * @brief get the owner- and overlap vertices for giving source and destination processes.
     *
     * The estimation is based on the vtxdist and the global PARMETIS mapping
     * generated before. The owner- and overlap vertices are stored in two
     * separate vectors
     *
     * @param graph The local graph.
     * @param part The target domain of the local vertices (result of PARMETIS).
     * @param indexSet The indexSet of the given graph.
     * @param parmetisVtxMapping The mapping between PARMETIS index
     *                           and DUNE global index.
     * @param myPe The source process number.
     * @param toPe The target process number.
     * @param ownerVec The output vector containing all owner vertices.
     * @param overlapSet The output vector containing all overlap vertices.
     */
    template<class OwnerSet, class G, class IS, class T, class GI>
    void getOwnerOverlapVec(const G& graph, std::vector<int>& part, IS& indexSet,
                            [[maybe_unused]] int myPe, int toPe, std::vector<T>& ownerVec, std::set<GI>& overlapSet,
                            RedistributeInterface& redist, std::set<int>& neighborProcs) {
      for(auto index = indexSet.begin(); index != indexSet.end(); ++index) {
        // Only Process owner vertices, the others are not in the parmetis graph.
        if(OwnerSet::contains(index->local().attribute()))
        {
          if(part[index->local()]==toPe)
          {
            getNeighbor<OwnerSet>(graph, part, index->local(), indexSet,
                                  toPe, overlapSet, neighborProcs);
            my_push_back(ownerVec, index->global(), toPe);
          }
        }
      }
      reserve(ownerVec, redist, toPe);

    }


    /**
     * @brief check if the given vertex is a owner vertex
     *
     * @param indexSet the indexSet
     * @param index the given vertex index
     */
    template<class F, class IS>
    inline bool isOwner(IS& indexSet, int index) {

      const typename IS::IndexPair* pindex=indexSet.pair(index);

      assert(pindex);
      return F::contains(pindex->local().attribute());
    }


    class BaseEdgeFunctor
    {
    public:
      BaseEdgeFunctor(Metis::idx_t* adj,const ParmetisDuneIndexMap& data)
        : i_(), adj_(adj), data_(data)
      {}

      template<class T>
      void operator()(const T& edge)
      {
        // Get the egde weight
        // const Weight& weight=edge.weight();
        adj_[i_] = data_.toParmetis(edge.target());
        i_++;
      }
      std::size_t index()
      {
        return i_;
      }

    private:
      std::size_t i_;
      Metis::idx_t* adj_;
      const ParmetisDuneIndexMap& data_;
    };

    template<typename G>
    struct EdgeFunctor
      : public BaseEdgeFunctor
    {
      EdgeFunctor(Metis::idx_t* adj, const ParmetisDuneIndexMap& data, std::size_t)
        : BaseEdgeFunctor(adj, data)
      {}

      Metis::idx_t* getWeights()
      {
        return NULL;
      }
      void free(){}
    };

    template<class G, class V, class E, class VM, class EM>
    class EdgeFunctor<Dune::Amg::PropertiesGraph<G,V,E,VM,EM> >
      :  public BaseEdgeFunctor
    {
    public:
      EdgeFunctor(Metis::idx_t* adj, const ParmetisDuneIndexMap& data, std::size_t s)
        : BaseEdgeFunctor(adj, data)
      {
        weight_=new Metis::idx_t[s];
      }

      template<class T>
      void operator()(const T& edge)
      {
        weight_[index()]=edge.properties().depends() ? 3 : 1;
        BaseEdgeFunctor::operator()(edge);
      }
      Metis::idx_t* getWeights()
      {
        return weight_;
      }
      void free(){
        if(weight_!=0) {
          delete weight_;
          weight_=0;
        }
      }
    private:
      Metis::idx_t* weight_;
    };



    /**
     * @brief Create the "adjncy" and "xadj" arrays for using ParMETIS
     *
     * This function builds the ParMETIS "adjncy" and "xadj" array according
     * to the ParMETIS documentation. These arrays are generated by
     * traversing the graph object. The assigned index to the
     * "adjncy" array is the ParMETIS global index calculated before.
     *
     * @param graph the local graph.
     * @param indexSet the local indexSet.
     * @param &xadj the ParMETIS xadj array
     * @param ew Funcot to setup adjacency info.
     */
    template<class F, class G, class IS, class EW>
    void getAdjArrays(G& graph, IS& indexSet, Metis::idx_t *xadj,
                      EW& ew)
    {
      int j=0;
      auto vend = graph.end();

      for(auto vertex = graph.begin(); vertex != vend; ++vertex) {
        if (isOwner<F>(indexSet,*vertex)) {
          // The type of const edge iterator.
          auto eend = vertex.end();
          xadj[j] = ew.index();
          j++;
          for(auto edge = vertex.begin(); edge != eend; ++edge) {
            ew(edge);
          }
        }
      }
      xadj[j] = ew.index();
    }
  } // end anonymous namespace

  template<class G, class T1, class T2>
  bool buildCommunication(const G& graph, std::vector<int>& realparts,
                          Dune::OwnerOverlapCopyCommunication<T1,T2>& oocomm,
                          std::shared_ptr<Dune::OwnerOverlapCopyCommunication<T1,T2>>& outcomm,
                          RedistributeInterface& redistInf,
                          bool verbose=false);
#if HAVE_PARMETIS
#ifndef METIS_VER_MAJOR
  extern "C"
  {
    // backwards compatibility to parmetis < 4.0.0
    void METIS_PartGraphKway(int *nvtxs, Metis::idx_t *xadj, Metis::idx_t *adjncy, Metis::idx_t *vwgt,
                             Metis::idx_t *adjwgt, int *wgtflag, int *numflag, int *nparts,
                             int *options, int *edgecut, Metis::idx_t *part);

    void METIS_PartGraphRecursive(int *nvtxs, Metis::idx_t *xadj, Metis::idx_t *adjncy, Metis::idx_t *vwgt,
                                  Metis::idx_t *adjwgt, int *wgtflag, int *numflag, int *nparts,
                                  int *options, int *edgecut, Metis::idx_t *part);
  }
#endif
#endif // HAVE_PARMETIS

  template<class S, class T>
  inline void print_carray(S& os, T* array, std::size_t l)
  {
    for(T *cur=array, *end=array+l; cur!=end; ++cur)
      os<<*cur<<" ";
  }

  template<class S, class T>
  inline bool isValidGraph(std::size_t noVtx, std::size_t gnoVtx, S noEdges, T* xadj,
                           T* adjncy, bool checkSymmetry)
  {
    bool correct=true;

    using std::signbit;
    for(Metis::idx_t vtx=0; vtx<(Metis::idx_t)noVtx; ++vtx) {
      if(static_cast<S>(xadj[vtx])>noEdges || signbit(xadj[vtx])) {
        std::cerr <<"Check graph: xadj["<<vtx<<"]="<<xadj[vtx]<<" (>"
                  <<noEdges<<") out of range!"<<std::endl;
        correct=false;
      }
      if(static_cast<S>(xadj[vtx+1])>noEdges || signbit(xadj[vtx+1])) {
        std::cerr <<"Check graph: xadj["<<vtx+1<<"]="<<xadj[vtx+1]<<" (>"
                  <<noEdges<<") out of range!"<<std::endl;
        correct=false;
      }
      // Check numbers in adjncy
      for(Metis::idx_t i=xadj[vtx]; i< xadj[vtx+1]; ++i) {
        if(signbit(adjncy[i]) || ((std::size_t)adjncy[i])>gnoVtx) {
          std::cerr<<" Edge "<<adjncy[i]<<" out of range ["<<0<<","<<noVtx<<")"
                   <<std::endl;
          correct=false;
        }
      }
      if(checkSymmetry) {
        for(Metis::idx_t i=xadj[vtx]; i< xadj[vtx+1]; ++i) {
          Metis::idx_t target=adjncy[i];
          // search for symmetric edge
          int found=0;
          for(Metis::idx_t j=xadj[target]; j< xadj[target+1]; ++j)
            if(adjncy[j]==vtx)
              found++;
          if(found!=1) {
            std::cerr<<"Edge ("<<target<<","<<vtx<<") "<<i<<" time"<<std::endl;
            correct=false;
          }
        }
      }
    }
    return correct;
  }

  template<class M, class T1, class T2>
  bool commGraphRepartition(const M& mat, Dune::OwnerOverlapCopyCommunication<T1,T2>& oocomm,
                            Metis::idx_t nparts,
                            std::shared_ptr<Dune::OwnerOverlapCopyCommunication<T1,T2>>& outcomm,
                            RedistributeInterface& redistInf,
                            bool verbose=false)
  {
    if(verbose && oocomm.communicator().rank()==0)
      std::cout<<"Repartitioning from "<<oocomm.communicator().size()
               <<" to "<<nparts<<" parts"<<std::endl;
    Timer time;
    int rank = oocomm.communicator().rank();
#if !HAVE_PARMETIS
    int* part = new int[1];
    part[0]=0;
#else
    Metis::idx_t* part = new Metis::idx_t[1]; // where all our data moves to

    if(nparts>1) {

      part[0]=rank;

      { // sublock for automatic memory deletion

        // Build the graph of the communication scheme and create an appropriate indexset.
        // calculate the neighbour vertices
        int noNeighbours = oocomm.remoteIndices().neighbours();

        for(auto n= oocomm.remoteIndices().begin(); n !=  oocomm.remoteIndices().end();
            ++n)
          if(n->first==rank) {
            //do not include ourselves.
            --noNeighbours;
            break;
          }

        // A parmetis graph representing the communication graph.
        // The diagonal entries are the number of nodes on the process.
        // The offdiagonal entries are the number of edges leading to other processes.

        Metis::idx_t *xadj=new Metis::idx_t[2];
        Metis::idx_t *vtxdist=new Metis::idx_t[oocomm.communicator().size()+1];
        Metis::idx_t *adjncy=new Metis::idx_t[noNeighbours];
#ifdef USE_WEIGHTS
        Metis::idx_t *vwgt = 0;
        Metis::idx_t *adjwgt = 0;
#endif

        // each process has exactly one vertex!
        for(int i=0; i<oocomm.communicator().size(); ++i)
          vtxdist[i]=i;
        vtxdist[oocomm.communicator().size()]=oocomm.communicator().size();

        xadj[0]=0;
        xadj[1]=noNeighbours;

        // count edges to other processor
        // a vector mapping the index to the owner
        // std::vector<int> owner(mat.N(), oocomm.communicator().rank());
        // for(NeighbourIterator n= oocomm.remoteIndices().begin(); n !=  oocomm.remoteIndices().end();
        //     ++n)
        //   {
        //     if(n->first!=oocomm.communicator().rank()){
        //       typedef typename RemoteIndices::RemoteIndexList RIList;
        //       const RIList& rlist = *(n->second.first);
        //       typedef typename RIList::const_iterator LIter;
        //       for(LIter entry=rlist.begin(); entry!=rlist.end(); ++entry){
        //         if(entry->attribute()==OwnerOverlapCopyAttributeSet::owner)
        //           owner[entry->localIndexPair().local()] = n->first;
        //       }
        //     }
        //   }

        // std::map<int,Metis::idx_t> edgecount; // edges to other processors
        // typedef typename M::ConstRowIterator RIter;
        // typedef typename M::ConstColIterator CIter;

        // // calculate edge count
        // for(RIter row=mat.begin(), endr=mat.end(); row != endr; ++row)
        //   if(owner[row.index()]==OwnerOverlapCopyAttributeSet::owner)
        //     for(CIter entry= row->begin(), ende = row->end(); entry != ende; ++entry)
        //       ++edgecount[owner[entry.index()]];

        // setup edge and weight pattern

        Metis::idx_t* adjp=adjncy;

#ifdef USE_WEIGHTS
        vwgt   = new Metis::idx_t[1];
        vwgt[0]= mat.N(); // weight is numer of rows TODO: Should actually be the nonzeros.

        adjwgt = new Metis::idx_t[noNeighbours];
        Metis::idx_t* adjwp=adjwgt;
#endif

        for(auto n= oocomm.remoteIndices().begin(); n !=  oocomm.remoteIndices().end();
            ++n)
          if(n->first != rank) {
            *adjp=n->first;
            ++adjp;
#ifdef USE_WEIGHTS
            *adjwp=1; //edgecount[n->first];
            ++adjwp;
#endif
          }
        assert(isValidGraph(vtxdist[rank+1]-vtxdist[rank],
                            vtxdist[oocomm.communicator().size()],
                            noNeighbours, xadj, adjncy, false));

        [[maybe_unused]] Metis::idx_t wgtflag=0;
        Metis::idx_t numflag=0;
        Metis::idx_t edgecut;
#ifdef USE_WEIGHTS
        wgtflag=3;
#endif
        Metis::real_t *tpwgts = new Metis::real_t[nparts];
        for(int i=0; i<nparts; ++i)
          tpwgts[i]=1.0/nparts;
        MPI_Comm comm=oocomm.communicator();

        Dune::dinfo<<rank<<" vtxdist: ";
        print_carray(Dune::dinfo, vtxdist, oocomm.communicator().size()+1);
        Dune::dinfo<<std::endl<<rank<<" xadj: ";
        print_carray(Dune::dinfo, xadj, 2);
        Dune::dinfo<<std::endl<<rank<<" adjncy: ";
        print_carray(Dune::dinfo, adjncy, noNeighbours);

#ifdef USE_WEIGHTS
        Dune::dinfo<<std::endl<<rank<<" vwgt: ";
        print_carray(Dune::dinfo, vwgt, 1);
        Dune::dinfo<<std::endl<<rank<<" adwgt: ";
        print_carray(Dune::dinfo, adjwgt, noNeighbours);
#endif
        Dune::dinfo<<std::endl;
        oocomm.communicator().barrier();
        if(verbose && oocomm.communicator().rank()==0)
          std::cout<<"Creating comm graph took "<<time.elapsed()<<std::endl;
        time.reset();

#ifdef PARALLEL_PARTITION
        Metis::real_t ubvec = 1.15;
        int ncon=1;
        int options[5] ={ 0,1,15,0,0};

        //=======================================================
        // ParMETIS_V3_PartKway
        //=======================================================
        ParMETIS_V3_PartKway(vtxdist, xadj, adjncy,
                             vwgt, adjwgt, &wgtflag,
                             &numflag, &ncon, &nparts, tpwgts, &ubvec, options, &edgecut, part,
                             &comm);
        if(verbose && oocomm.communicator().rank()==0)
          std::cout<<"ParMETIS took "<<time.elapsed()<<std::endl;
        time.reset();
#else
        Timer time1;
        std::size_t gnoedges=0;
        int* noedges = 0;
        noedges = new int[oocomm.communicator().size()];
        Dune::dverb<<"noNeighbours: "<<noNeighbours<<std::endl;
        // gather number of edges for each vertex.
        MPI_Allgather(&noNeighbours,1,MPI_INT,noedges,1, MPI_INT,oocomm.communicator());

        if(verbose && oocomm.communicator().rank()==0)
          std::cout<<"Gathering noedges took "<<time1.elapsed()<<std::endl;
        time1.reset();

        Metis::idx_t noVertices = vtxdist[oocomm.communicator().size()];
        Metis::idx_t *gxadj = 0;
        Metis::idx_t *gvwgt = 0;
        Metis::idx_t *gadjncy = 0;
        Metis::idx_t *gadjwgt = 0;
        Metis::idx_t *gpart = 0;
        int* displ = 0;
        int* noxs = 0;
        int* xdispl = 0;  // displacement for xadj
        int* novs = 0;
        int* vdispl=0; // real vertex displacement
#ifdef USE_WEIGHTS
        std::size_t localNoVtx=vtxdist[rank+1]-vtxdist[rank];
#endif
        std::size_t gxadjlen = vtxdist[oocomm.communicator().size()]-vtxdist[0]+oocomm.communicator().size();

        {
          Dune::dinfo<<"noedges: ";
          print_carray(Dune::dinfo, noedges, oocomm.communicator().size());
          Dune::dinfo<<std::endl;
          displ = new int[oocomm.communicator().size()];
          xdispl = new int[oocomm.communicator().size()];
          noxs = new int[oocomm.communicator().size()];
          vdispl = new int[oocomm.communicator().size()];
          novs = new int[oocomm.communicator().size()];

          for(int i=0; i < oocomm.communicator().size(); ++i) {
            noxs[i]=vtxdist[i+1]-vtxdist[i]+1;
            novs[i]=vtxdist[i+1]-vtxdist[i];
          }

          Metis::idx_t *so= vtxdist;
          int offset = 0;
          for(int *xcurr = xdispl, *vcurr = vdispl, *end=vdispl+oocomm.communicator().size();
              vcurr!=end; ++vcurr, ++xcurr, ++so, ++offset) {
            *vcurr = *so;
            *xcurr = offset + *so;
          }

          int *pdispl =displ;
          int cdispl = 0;
          *pdispl = 0;
          for(int *curr=noedges, *end=noedges+oocomm.communicator().size()-1;
              curr!=end; ++curr) {
            ++pdispl; // next displacement
            cdispl += *curr; // next value
            *pdispl = cdispl;
          }
          Dune::dinfo<<"displ: ";
          print_carray(Dune::dinfo, displ, oocomm.communicator().size());
          Dune::dinfo<<std::endl;

          // calculate global number of edges
          // It is bigger than the actual one as we habe size-1 additional end entries
          for(int *curr=noedges, *end=noedges+oocomm.communicator().size();
              curr!=end; ++curr)
            gnoedges += *curr;

          // alocate gobal graph
          Dune::dinfo<<"gxadjlen: "<<gxadjlen<<" noVertices: "<<noVertices
                     <<" gnoedges: "<<gnoedges<<std::endl;
          gxadj = new Metis::idx_t[gxadjlen];
          gpart = new Metis::idx_t[noVertices];
#ifdef USE_WEIGHTS
          gvwgt = new Metis::idx_t[noVertices];
          gadjwgt = new Metis::idx_t[gnoedges];
#endif
          gadjncy = new Metis::idx_t[gnoedges];
        }

        if(verbose && oocomm.communicator().rank()==0)
          std::cout<<"Preparing global graph took "<<time1.elapsed()<<std::endl;
        time1.reset();
        // Communicate data

        MPI_Allgatherv(xadj,2,MPITraits<Metis::idx_t>::getType(),
                       gxadj,noxs,xdispl,MPITraits<Metis::idx_t>::getType(),
                       comm);
        MPI_Allgatherv(adjncy,noNeighbours,MPITraits<Metis::idx_t>::getType(),
                       gadjncy,noedges,displ,MPITraits<Metis::idx_t>::getType(),
                       comm);
#ifdef USE_WEIGHTS
        MPI_Allgatherv(adjwgt,noNeighbours,MPITraits<Metis::idx_t>::getType(),
                       gadjwgt,noedges,displ,MPITraits<Metis::idx_t>::getType(),
                       comm);
        MPI_Allgatherv(vwgt,localNoVtx,MPITraits<Metis::idx_t>::getType(),
                       gvwgt,novs,vdispl,MPITraits<Metis::idx_t>::getType(),
                       comm);
#endif
        if(verbose && oocomm.communicator().rank()==0)
          std::cout<<"Gathering global graph data took "<<time1.elapsed()<<std::endl;
        time1.reset();

        {
          // create the real gxadj array
          // i.e. shift entries and add displacements.

          print_carray(Dune::dinfo, gxadj, gxadjlen);

          int offset = 0;
          Metis::idx_t increment = vtxdist[1];
          Metis::idx_t *start=gxadj+1;
          for(int i=1; i<oocomm.communicator().size(); ++i) {
            offset+=1;
            int lprev = vtxdist[i]-vtxdist[i-1];
            int l = vtxdist[i+1]-vtxdist[i];
            start+=lprev;
            assert((start+l+offset)-gxadj<=static_cast<Metis::idx_t>(gxadjlen));
            increment = *(start-1);
            std::transform(start+offset, start+l+offset, start, std::bind(std::plus<Metis::idx_t>(), std::placeholders::_1, increment));
          }
          Dune::dinfo<<std::endl<<"shifted xadj:";
          print_carray(Dune::dinfo, gxadj, noVertices+1);
          Dune::dinfo<<std::endl<<" gadjncy: ";
          print_carray(Dune::dinfo, gadjncy, gnoedges);
#ifdef USE_WEIGHTS
          Dune::dinfo<<std::endl<<" gvwgt: ";
          print_carray(Dune::dinfo, gvwgt, noVertices);
          Dune::dinfo<<std::endl<<"adjwgt: ";
          print_carray(Dune::dinfo, gadjwgt, gnoedges);
          Dune::dinfo<<std::endl;
#endif
          // everything should be fine now!!!
          if(verbose && oocomm.communicator().rank()==0)
            std::cout<<"Postprocesing global graph data took "<<time1.elapsed()<<std::endl;
          time1.reset();
#ifndef NDEBUG
          assert(isValidGraph(noVertices, noVertices, gnoedges,
                              gxadj, gadjncy, true));
#endif

          if(verbose && oocomm.communicator().rank()==0)
            std::cout<<"Creating grah one 1 process took "<<time.elapsed()<<std::endl;
          time.reset();
#if METIS_VER_MAJOR >= 5
          Metis::idx_t ncon = 1;
          Metis::idx_t moptions[METIS_NOPTIONS];
          METIS_SetDefaultOptions(moptions);
          moptions[METIS_OPTION_NUMBERING] = numflag;
          METIS_PartGraphRecursive(&noVertices, &ncon, gxadj, gadjncy, gvwgt, NULL, gadjwgt,
                         &nparts, NULL, NULL, moptions, &edgecut, gpart);
#else
          int options[5] = {0, 1, 1, 3, 3};
          // Call metis
          METIS_PartGraphRecursive(&noVertices, gxadj, gadjncy, gvwgt, gadjwgt, &wgtflag,
                                   &numflag, &nparts, options, &edgecut, gpart);
#endif

          if(verbose && oocomm.communicator().rank()==0)
            std::cout<<"METIS took "<<time.elapsed()<<std::endl;
          time.reset();

          Dune::dinfo<<std::endl<<"part:";
          print_carray(Dune::dinfo, gpart, noVertices);

          delete[] gxadj;
          delete[] gadjncy;
#ifdef USE_WEIGHTS
          delete[] gvwgt;
          delete[] gadjwgt;
#endif
        }
        // Scatter result
        MPI_Scatter(gpart, 1, MPITraits<Metis::idx_t>::getType(), part, 1,
                    MPITraits<Metis::idx_t>::getType(), 0, comm);

        {
          // release remaining memory
          delete[] gpart;
          delete[] noedges;
          delete[] displ;
        }


#endif
        delete[] xadj;
        delete[] vtxdist;
        delete[] adjncy;
#ifdef USE_WEIGHTS
        delete[] vwgt;
        delete[] adjwgt;
#endif
        delete[] tpwgts;
      }
    }else{
      part[0]=0;
    }
#endif
    Dune::dinfo<<" repart "<<rank <<" -> "<< part[0]<<std::endl;

    std::vector<int> realpart(mat.N(), part[0]);
    delete[] part;

    oocomm.copyOwnerToAll(realpart, realpart);

    if(verbose && oocomm.communicator().rank()==0)
      std::cout<<"Scattering repartitioning took "<<time.elapsed()<<std::endl;
    time.reset();


    oocomm.buildGlobalLookup(mat.N());
    Dune::Amg::MatrixGraph<M> graph(const_cast<M&>(mat));
    fillIndexSetHoles(graph, oocomm);
    if(verbose && oocomm.communicator().rank()==0)
      std::cout<<"Filling index set took "<<time.elapsed()<<std::endl;
    time.reset();

    if(verbose) {
      int noNeighbours=oocomm.remoteIndices().neighbours();
      noNeighbours = oocomm.communicator().sum(noNeighbours)
                     / oocomm.communicator().size();
      if(oocomm.communicator().rank()==0)
        std::cout<<"Average no neighbours was "<<noNeighbours<<std::endl;
    }
    bool ret = buildCommunication(graph, realpart, oocomm, outcomm, redistInf,
                                  verbose);
    if(verbose && oocomm.communicator().rank()==0)
      std::cout<<"Building index sets took "<<time.elapsed()<<std::endl;
    time.reset();


    return ret;

  }

  /**
   * @brief execute a graph repartition for a giving graph and indexset.
   *
   * This function provides repartition functionality using the
   * PARMETIS library
   *
   * @param graph The given graph to repartition
   * @param oocomm The parallel information about the graph.
   * @param nparts The number of domains the repartitioning should achieve.
   * @param[out] outcomm Pointer store the parallel information of the
   * redistributed domains in.
   * @param redistInf Redistribute interface
   * @param verbose Verbosity flag to give out additional information.
   */
  template<class G, class T1, class T2>
  bool graphRepartition(const G& graph, Dune::OwnerOverlapCopyCommunication<T1,T2>& oocomm, Metis::idx_t nparts,
                        std::shared_ptr<Dune::OwnerOverlapCopyCommunication<T1,T2>>& outcomm,
                        RedistributeInterface& redistInf,
                        bool verbose=false)
  {
    Timer time;

    MPI_Comm comm=oocomm.communicator();
    oocomm.buildGlobalLookup(graph.noVertices());
    fillIndexSetHoles(graph, oocomm);

    if(verbose && oocomm.communicator().rank()==0)
      std::cout<<"Filling holes took "<<time.elapsed()<<std::endl;
    time.reset();

    // simple precondition checks

#ifdef PERF_REPART
    // Profiling variables
    double t1=0.0, t2=0.0, t3=0.0, t4=0.0, tSum=0.0;
#endif


    // MPI variables
    int mype = oocomm.communicator().rank();

    assert(nparts<=static_cast<Metis::idx_t>(oocomm.communicator().size()));

    int myDomain = -1;

    //
    // 1) Prepare the required parameters for using ParMETIS
    //    Especially the arrays that represent the graph must be
    //    generated by the DUNE Graph and IndexSet input variables.
    //    These are the arrays:
    //    - vtxdist
    //    - xadj
    //    - adjncy
    //
    //
#ifdef PERF_REPART
    // reset timer for step 1)
    t1=MPI_Wtime();
#endif


    typedef typename  Dune::OwnerOverlapCopyCommunication<T1,T2> OOComm;
    typedef typename  OOComm::OwnerSet OwnerSet;

    // Create the vtxdist array and parmetisVtxMapping.
    // Global communications are necessary
    // The parmetis global identifiers for the owner vertices.
    ParmetisDuneIndexMap indexMap(graph,oocomm);
    Metis::idx_t *part = new Metis::idx_t[indexMap.numOfOwnVtx()];
    for(std::size_t i=0; i < indexMap.numOfOwnVtx(); ++i)
      part[i]=mype;

#if !HAVE_PARMETIS
    if(oocomm.communicator().rank()==0 && nparts>1)
      std::cerr<<"ParMETIS not activated. Will repartition to 1 domain instead of requested "
               <<nparts<<" domains."<<std::endl;
    nparts=1; // No parmetis available, fallback to agglomerating to 1 process

#else

    if(nparts>1) {
      // Create the xadj and adjncy arrays
      Metis::idx_t *xadj = new  Metis::idx_t[indexMap.numOfOwnVtx()+1];
      Metis::idx_t *adjncy = new Metis::idx_t[graph.noEdges()];
      EdgeFunctor<G> ef(adjncy, indexMap, graph.noEdges());
      getAdjArrays<OwnerSet>(graph, oocomm.globalLookup(), xadj, ef);

      //
      // 2) Call ParMETIS
      //
      //
      Metis::idx_t numflag=0, wgtflag=0, options[3], edgecut=0, ncon=1;
      //float *tpwgts = NULL;
      Metis::real_t *tpwgts = new Metis::real_t[nparts];
      for(int i=0; i<nparts; ++i)
        tpwgts[i]=1.0/nparts;
      Metis::real_t ubvec[1];
      options[0] = 0; // 0=default, 1=options are defined in [1]+[2]
#ifdef DEBUG_REPART
      options[1] = 3; // show info: 0=no message
#else
      options[1] = 0; // show info: 0=no message
#endif
      options[2] = 1; // random number seed, default is 15
      wgtflag = (ef.getWeights()!=NULL) ? 1 : 0;
      numflag = 0;
      edgecut = 0;
      ncon=1;
      ubvec[0]=1.05; // recommended by ParMETIS

#ifdef DEBUG_REPART
      if (mype == 0) {
        std::cout<<std::endl;
        std::cout<<"Testing ParMETIS_V3_PartKway with options[1-2] = {"
                 <<options[1]<<" "<<options[2]<<"}, Ncon: "
                 <<ncon<<", Nparts: "<<nparts<<std::endl;
      }
#endif
#ifdef PERF_REPART
      // stop the time for step 1)
      t1=MPI_Wtime()-t1;
      // reset timer for step 2)
      t2=MPI_Wtime();
#endif

      if(verbose) {
        oocomm.communicator().barrier();
        if(oocomm.communicator().rank()==0)
          std::cout<<"Preparing for parmetis took "<<time.elapsed()<<std::endl;
      }
      time.reset();

      //=======================================================
      // ParMETIS_V3_PartKway
      //=======================================================
      ParMETIS_V3_PartKway(indexMap.vtxDist(), xadj, adjncy,
                           NULL, ef.getWeights(), &wgtflag,
                           &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &const_cast<MPI_Comm&>(comm));


      delete[] xadj;
      delete[] adjncy;
      delete[] tpwgts;

      ef.free();

#ifdef DEBUG_REPART
      if (mype == 0) {
        std::cout<<std::endl;
        std::cout<<"ParMETIS_V3_PartKway reported a cut of "<<edgecut<<std::endl;
        std::cout<<std::endl;
      }
      std::cout<<mype<<": PARMETIS-Result: ";
      for(int i=0; i < indexMap.vtxDist()[mype+1]-indexMap.vtxDist()[mype]; ++i) {
        std::cout<<part[i]<<" ";
      }
      std::cout<<std::endl;
      std::cout<<"Testing ParMETIS_V3_PartKway with options[1-2] = {"
               <<options[1]<<" "<<options[2]<<"}, Ncon: "
               <<ncon<<", Nparts: "<<nparts<<std::endl;
#endif
#ifdef PERF_REPART
      // stop the time for step 2)
      t2=MPI_Wtime()-t2;
      // reset timer for step 3)
      t3=MPI_Wtime();
#endif


      if(verbose) {
        oocomm.communicator().barrier();
        if(oocomm.communicator().rank()==0)
          std::cout<<"Parmetis took "<<time.elapsed()<<std::endl;
      }
      time.reset();
    }else
#endif
    {
      // Everything goes to process 0!
      for(std::size_t i=0; i<indexMap.numOfOwnVtx(); ++i)
        part[i]=0;
    }


    //
    // 3) Find a optimal domain based on the ParMETIS repartitioning
    //    result
    //

    std::vector<int> domainMapping(nparts);
    if(nparts>1)
      getDomain(comm, part, indexMap.numOfOwnVtx(), nparts, &myDomain, domainMapping);
    else
      domainMapping[0]=0;

#ifdef DEBUG_REPART
    std::cout<<mype<<": myDomain: "<<myDomain<<std::endl;
    std::cout<<mype<<": DomainMapping: ";
    for(auto j : range(nparts)) {
      std::cout<<" do: "<<j<<" pe: "<<domainMapping[j]<<" ";
    }
    std::cout<<std::endl;
#endif

    // Make a domain mapping for the indexset and translate
    //domain number to real process number
    // domainMapping is the one of parmetis, that is without
    // the overlap/copy vertices
    std::vector<int> setPartition(oocomm.indexSet().size(), -1);

    std::size_t i=0; // parmetis index
    for(auto index = oocomm.indexSet().begin(); index != oocomm.indexSet().end(); ++index)
      if(OwnerSet::contains(index->local().attribute())) {
        setPartition[index->local()]=domainMapping[part[i++]];
      }

    delete[] part;
    oocomm.copyOwnerToAll(setPartition, setPartition);
    // communication only needed for ALU
    // (ghosts with same global id as owners on the same process)
    if (SolverCategory::category(oocomm) ==
        static_cast<int>(SolverCategory::nonoverlapping))
      oocomm.copyCopyToAll(setPartition, setPartition);
    bool ret = buildCommunication(graph, setPartition, oocomm, outcomm, redistInf,
                                  verbose);
    if(verbose) {
      oocomm.communicator().barrier();
      if(oocomm.communicator().rank()==0)
        std::cout<<"Creating indexsets took "<<time.elapsed()<<std::endl;
    }
    return ret;
  }



  template<class G, class T1, class T2>
  bool buildCommunication(const G& graph,
                          std::vector<int>& setPartition, Dune::OwnerOverlapCopyCommunication<T1,T2>& oocomm,
                          std::shared_ptr<Dune::OwnerOverlapCopyCommunication<T1,T2>>& outcomm,
                          RedistributeInterface& redistInf,
                          bool verbose)
  {
    typedef typename  Dune::OwnerOverlapCopyCommunication<T1,T2> OOComm;
    typedef typename  OOComm::OwnerSet OwnerSet;

    Timer time;

    // Build the send interface
    redistInf.buildSendInterface<OwnerSet>(setPartition, oocomm.indexSet());

#ifdef PERF_REPART
    // stop the time for step 3)
    t3=MPI_Wtime()-t3;
    // reset timer for step 4)
    t4=MPI_Wtime();
#endif


    //
    // 4) Create the output IndexSet and RemoteIndices
    //    4.1) Determine the "send to" and "receive from" relation
    //         according to the new partition using a MPI ring
    //         communication.
    //
    //    4.2) Depends on the "send to" and "receive from" vector,
    //         the processes will exchange the vertices each other
    //
    //    4.3) Create the IndexSet, RemoteIndices and the new MPI
    //         communicator
    //

    //
    // 4.1) Let's start...
    //
    int npes = oocomm.communicator().size();
    int *sendTo = 0;
    int noSendTo = 0;
    std::set<int> recvFrom;

    // the max number of vertices is stored in the sendTo buffer,
    // not the number of vertices to send! Because the max number of Vtx
    // is used as the fixed buffer size by the MPI send/receive calls

    int mype = oocomm.communicator().rank();

    {
      std::set<int> tsendTo;
      for(auto i=setPartition.begin(), iend = setPartition.end(); i!=iend; ++i)
        tsendTo.insert(*i);

      noSendTo = tsendTo.size();
      sendTo = new int[noSendTo];
      int idx=0;
      for(auto i=tsendTo.begin(); i != tsendTo.end(); ++i, ++idx)
        sendTo[idx]=*i;
    }

    //
    int* gnoSend= new int[oocomm.communicator().size()];
    int* gsendToDispl =  new int[oocomm.communicator().size()+1];

    MPI_Allgather(&noSendTo, 1, MPI_INT, gnoSend, 1,
                  MPI_INT, oocomm.communicator());

    // calculate total receive message size
    int totalNoRecv = 0;
    for(int i=0; i<npes; ++i)
      totalNoRecv += gnoSend[i];

    int *gsendTo = new int[totalNoRecv];

    // calculate displacement for allgatherv
    gsendToDispl[0]=0;
    for(int i=0; i<npes; ++i)
      gsendToDispl[i+1]=gsendToDispl[i]+gnoSend[i];

    // gather the data
    MPI_Allgatherv(sendTo, noSendTo, MPI_INT, gsendTo, gnoSend, gsendToDispl,
                   MPI_INT, oocomm.communicator());

    // Extract from which processes we will receive data
    for(int proc=0; proc < npes; ++proc)
      for(int i=gsendToDispl[proc]; i < gsendToDispl[proc+1]; ++i)
        if(gsendTo[i]==mype)
          recvFrom.insert(proc);

    bool existentOnNextLevel = recvFrom.size()>0;

    // Delete memory
    delete[] gnoSend;
    delete[] gsendToDispl;
    delete[] gsendTo;


#ifdef DEBUG_REPART
    if(recvFrom.size()) {
      std::cout<<mype<<": recvFrom: ";
      for(auto i=recvFrom.begin(); i!= recvFrom.end(); ++i) {
        std::cout<<*i<<" ";
      }
    }

    std::cout<<std::endl<<std::endl;
    std::cout<<mype<<": sendTo: ";
    for(int i=0; i<noSendTo; i++) {
      std::cout<<sendTo[i]<<" ";
    }
    std::cout<<std::endl<<std::endl;
#endif

    if(verbose)
      if(oocomm.communicator().rank()==0)
        std::cout<<" Communicating the receive information took "<<
        time.elapsed()<<std::endl;
    time.reset();

    //
    // 4.2) Start the communication
    //

    // Get all the owner and overlap vertices for myself ans save
    // it in the vectors myOwnerVec and myOverlapVec.
    // The received vertices from the other processes are simple
    // added to these vector.
    //


    typedef typename OOComm::ParallelIndexSet::GlobalIndex GI;
    typedef std::vector<GI> GlobalVector;
    std::vector<std::pair<GI,int> > myOwnerVec;
    std::set<GI> myOverlapSet;
    GlobalVector sendOwnerVec;
    std::set<GI> sendOverlapSet;
    std::set<int> myNeighbors;

    //    getOwnerOverlapVec<OwnerSet>(graph, setPartition, oocomm.globalLookup(),
    //				 mype, mype, myOwnerVec, myOverlapSet, redistInf, myNeighbors);

    char **sendBuffers=new char*[noSendTo];
    MPI_Request *requests = new MPI_Request[noSendTo];

    // Create all messages to be sent
    for(int i=0; i < noSendTo; ++i) {
      // clear the vector for sending
      sendOwnerVec.clear();
      sendOverlapSet.clear();
      // get all owner and overlap vertices for process j and save these
      // in the vectors sendOwnerVec and sendOverlapSet
      std::set<int> neighbors;
      getOwnerOverlapVec<OwnerSet>(graph, setPartition, oocomm.globalLookup(),
                                   mype, sendTo[i], sendOwnerVec, sendOverlapSet, redistInf,
                                   neighbors);
      // +2, we need 2 integer more for the length of each part
      // (owner/overlap) of the array
      int buffersize=0;
      int tsize;
      MPI_Pack_size(1, MPITraits<std::size_t>::getType(), oocomm.communicator(), &buffersize);
      MPI_Pack_size(sendOwnerVec.size(), MPITraits<GI>::getType(), oocomm.communicator(), &tsize);
      buffersize +=tsize;
      MPI_Pack_size(1, MPITraits<std::size_t>::getType(), oocomm.communicator(), &tsize);
      buffersize +=tsize;
      MPI_Pack_size(sendOverlapSet.size(), MPITraits<GI>::getType(), oocomm.communicator(), &tsize);
      buffersize += tsize;
      MPI_Pack_size(1, MPITraits<std::size_t>::getType(), oocomm.communicator(), &tsize);
      buffersize += tsize;
      MPI_Pack_size(neighbors.size(), MPI_INT, oocomm.communicator(), &tsize);
      buffersize += tsize;

      sendBuffers[i] = new char[buffersize];

#ifdef DEBUG_REPART
      std::cout<<mype<<" sending "<<sendOwnerVec.size()<<" owner and "<<
      sendOverlapSet.size()<<" overlap to "<<sendTo[i]<<" buffersize="<<buffersize<<std::endl;
#endif
      createSendBuf(sendOwnerVec, sendOverlapSet, neighbors, sendBuffers[i], buffersize, oocomm.communicator());
      MPI_Issend(sendBuffers[i], buffersize, MPI_PACKED, sendTo[i], 99, oocomm.communicator(), requests+i);
    }

    if(verbose) {
      oocomm.communicator().barrier();
      if(oocomm.communicator().rank()==0)
        std::cout<<" Creating sends took "<<
        time.elapsed()<<std::endl;
    }
    time.reset();

    // Receive Messages
    int noRecv = recvFrom.size();
    int oldbuffersize=0;
    char* recvBuf = 0;
    while(noRecv>0) {
      // probe for an incoming message
      MPI_Status stat;
      MPI_Probe(MPI_ANY_SOURCE, 99,  oocomm.communicator(), &stat);
      int buffersize;
      MPI_Get_count(&stat, MPI_PACKED, &buffersize);

      if(oldbuffersize<buffersize) {
        // buffer too small, reallocate
        delete[] recvBuf;
        recvBuf = new char[buffersize];
        oldbuffersize = buffersize;
      }
      MPI_Recv(recvBuf, buffersize, MPI_PACKED, stat.MPI_SOURCE, 99, oocomm.communicator(), &stat);
      saveRecvBuf(recvBuf, buffersize, myOwnerVec, myOverlapSet, myNeighbors, redistInf,
                  stat.MPI_SOURCE, oocomm.communicator());
      --noRecv;
    }

    if(recvBuf)
      delete[] recvBuf;

    time.reset();
    // Wait for sending messages to complete
    MPI_Status *statuses = new MPI_Status[noSendTo];
    int send = MPI_Waitall(noSendTo, requests, statuses);

    // check for errors
    if(send==MPI_ERR_IN_STATUS) {
      std::cerr<<mype<<": Error in sending :"<<std::endl;
      // Search for the error
      for(int i=0; i< noSendTo; i++)
        if(statuses[i].MPI_ERROR!=MPI_SUCCESS) {
          char message[300];
          int messageLength;
          MPI_Error_string(statuses[i].MPI_ERROR, message, &messageLength);
          std::cerr<<" source="<<statuses[i].MPI_SOURCE<<" message: ";
          for(int j = 0; j < messageLength; j++)
            std::cout<<message[j];
        }
      std::cerr<<std::endl;
    }

    if(verbose) {
      oocomm.communicator().barrier();
      if(oocomm.communicator().rank()==0)
        std::cout<<" Receiving and saving took "<<
        time.elapsed()<<std::endl;
    }
    time.reset();

    for(int i=0; i < noSendTo; ++i)
      delete[] sendBuffers[i];

    delete[] sendBuffers;
    delete[] statuses;
    delete[] requests;

    redistInf.setCommunicator(oocomm.communicator());

    //
    // 4.2) Create the IndexSet etc.
    //

    // build the new outputIndexSet


    int color=0;

    if (!existentOnNextLevel) {
      // this process is not used anymore
      color= MPI_UNDEFINED;
    }
    MPI_Comm outputComm;

    MPI_Comm_split(oocomm.communicator(), color, oocomm.communicator().rank(), &outputComm);
    outcomm = std::make_shared<OOComm>(outputComm,SolverCategory::category(oocomm),true);

    // translate neighbor ranks.
    int newrank=outcomm->communicator().rank();
    int *newranks=new int[oocomm.communicator().size()];
    std::vector<int> tneighbors;
    tneighbors.reserve(myNeighbors.size());

    typename OOComm::ParallelIndexSet& outputIndexSet = outcomm->indexSet();

    MPI_Allgather(&newrank, 1, MPI_INT, newranks, 1,
                  MPI_INT, oocomm.communicator());

#ifdef DEBUG_REPART
    std::cout<<oocomm.communicator().rank()<<" ";
    for(auto i=myNeighbors.begin(), end=myNeighbors.end();
        i!=end; ++i) {
      assert(newranks[*i]>=0);
      std::cout<<*i<<"->"<<newranks[*i]<<" ";
      tneighbors.push_back(newranks[*i]);
    }
    std::cout<<std::endl;
#else
    for(auto i=myNeighbors.begin(), end=myNeighbors.end();
        i!=end; ++i) {
      tneighbors.push_back(newranks[*i]);
    }
#endif
    delete[] newranks;
    myNeighbors.clear();

    if(verbose) {
      oocomm.communicator().barrier();
      if(oocomm.communicator().rank()==0)
        std::cout<<" Calculating new neighbours ("<<tneighbors.size()<<") took "<<
        time.elapsed()<<std::endl;
    }
    time.reset();


    outputIndexSet.beginResize();
    // 1) add the owner vertices
    // Sort the owners
    std::sort(myOwnerVec.begin(), myOwnerVec.end(), SortFirst());
    // The owners are sorted according to there global index
    // Therefore the entries of ownerVec are the same as the
    // ones in the resulting index set.
    int i=0;
    using LocalIndexT = typename OOComm::ParallelIndexSet::LocalIndex;
    for(auto g=myOwnerVec.begin(), end =myOwnerVec.end(); g!=end; ++g, ++i ) {
      outputIndexSet.add(g->first,LocalIndexT(i, OwnerOverlapCopyAttributeSet::owner, true));
      redistInf.addReceiveIndex(g->second, i);
    }

    if(verbose) {
      oocomm.communicator().barrier();
      if(oocomm.communicator().rank()==0)
        std::cout<<" Adding owner indices took "<<
        time.elapsed()<<std::endl;
    }
    time.reset();


    // After all the vertices are received, the vectors must
    // be "merged" together to create the final vectors.
    // Because some vertices that are sent as overlap could now
    // already included as owner vertiecs in the new partition
    mergeVec(myOwnerVec, myOverlapSet);

    // Trick to free memory
    myOwnerVec.clear();
    myOwnerVec.swap(myOwnerVec);

    if(verbose) {
      oocomm.communicator().barrier();
      if(oocomm.communicator().rank()==0)
        std::cout<<" Merging indices took "<<
        time.elapsed()<<std::endl;
    }
    time.reset();


    // 2) add the overlap vertices
    for(auto g=myOverlapSet.begin(), end=myOverlapSet.end(); g!=end; ++g, i++) {
      outputIndexSet.add(*g,LocalIndexT(i, OwnerOverlapCopyAttributeSet::copy, true));
    }
    myOverlapSet.clear();
    outputIndexSet.endResize();

#ifdef DUNE_ISTL_WITH_CHECKING
    int numOfOwnVtx =0;
    auto end = outputIndexSet.end();
    for(auto index = outputIndexSet.begin(); index != end; ++index) {
      if (OwnerSet::contains(index->local().attribute())) {
        numOfOwnVtx++;
      }
    }
    numOfOwnVtx = oocomm.communicator().sum(numOfOwnVtx);
    // if(numOfOwnVtx!=indexMap.globalOwnerVertices)
    //   {
    //     std::cerr<<numOfOwnVtx<<"!="<<indexMap.globalOwnerVertices<<" owners missing or additional ones!"<<std::endl;
    //     DUNE_THROW(ISTLError, numOfOwnVtx<<"!="<<indexMap.globalOwnerVertices<<" owners missing or additional ones"
    //             <<" during repartitioning.");
    //   }
    std::is_sorted(outputIndexSet.begin(), outputIndexSet.end(),
                   [](const auto& v1, const auto& v2){ return v1.global() < v2.global();});
#endif
    if(verbose) {
      oocomm.communicator().barrier();
      if(oocomm.communicator().rank()==0)
        std::cout<<" Adding overlap indices took "<<
        time.elapsed()<<std::endl;
    }
    time.reset();


    if(color != MPI_UNDEFINED) {
      outcomm->remoteIndices().setNeighbours(tneighbors);
      outcomm->remoteIndices().template rebuild<true>();

    }

    // release the memory
    delete[] sendTo;

    if(verbose) {
      oocomm.communicator().barrier();
      if(oocomm.communicator().rank()==0)
        std::cout<<" Storing indexsets took "<<
        time.elapsed()<<std::endl;
    }

#ifdef PERF_REPART
    // stop the time for step 4) and print the results
    t4=MPI_Wtime()-t4;
    tSum = t1 + t2 + t3 + t4;
    std::cout<<std::endl
             <<mype<<": WTime for step 1): "<<t1
             <<" 2): "<<t2
             <<" 3): "<<t3
             <<" 4): "<<t4
             <<" total: "<<tSum
             <<std::endl;
#endif

    return color!=MPI_UNDEFINED;

  }
#else
  template<class G, class P,class T1, class T2, class R>
  bool graphRepartition(const G& graph, P& oocomm, int nparts,
                        std::shared_ptr<P>& outcomm,
                        R& redistInf,
                        bool v=false)
  {
    if(nparts!=oocomm.size())
      DUNE_THROW(NotImplemented, "only available for MPI programs");
  }


  template<class G, class P,class T1, class T2, class R>
  bool commGraphRepartition(const G& graph, P& oocomm, int nparts,
                            std::shared_ptr<P>& outcomm,
                            R& redistInf,
                            bool v=false)
  {
    if(nparts!=oocomm.size())
      DUNE_THROW(NotImplemented, "only available for MPI programs");
  }
#endif // HAVE_MPI
} // end of namespace Dune
#endif
