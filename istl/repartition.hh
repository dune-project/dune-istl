// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_REPARTITION_HH
#define DUNE_REPARTITION_HH

#if HAVE_PARMETIS
#include <parmetis.h>
#endif
#include "config.h"
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/mpitraits.hh>
#include <dune/istl/indexset.hh>
#include <dune/istl/remoteindices.hh>
#include <dune/istl/indicessyncer.hh>
#include <dune/istl/communicator.hh>
#include <dune/common/enumset.hh>
#include <map>
#include <utility>
#include <cassert>

/**
 * @file
 * @brief Functionality for redistributing a parallel index set using graph partitioning.
 *
 * Refactored version of an intern.
 * @author Markus Blatt
 */

namespace Dune
{
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
    IndexSet& indexSet = oocomm.indexSet();
    const typename Dune::OwnerOverlapCopyCommunication<T1,T2>::GlobalLookupIndexSet& lookup =oocomm.globalLookup();

    // The type of the const vertex iterator.
    typedef typename G::ConstVertexIterator VertexIterator;


    std::size_t sum=0, needed = graph.noVertices()-indexSet.size();
    std::vector<std::size_t> neededall(oocomm.communicator().size(), 0);

    MPI_Allgather(&needed, 1, Generic_MPI_Datatype<std::size_t>::get() , &(neededall[0]), 1, Generic_MPI_Datatype<std::size_t>::get(), oocomm.communicator());
    for(int i=0; i<oocomm.communicator().size(); ++i)
      sum=sum+neededall[i];   // MAke this for generic

    if(sum==0)
      // Nothing to do
      return;

    //Compute Maximum Global Index
    T1 maxgi=0;
    typedef typename IndexSet::const_iterator Iterator;
    Iterator end;
    end = indexSet.end();
    for(Iterator it = indexSet.begin(); it != end; ++it)
      maxgi=std::max(maxgi,it->global());

    //Process p creates global indices consecutively
    //starting atmaxgi+\sum_{i=1}^p neededall[i]
    // All created indices are owned by the process
    maxgi=oocomm.communicator().max(maxgi);
    ++maxgi;  //Sart with the next free index.

    for(int i=0; i<oocomm.communicator().rank(); ++i)
      maxgi=maxgi+neededall[i];   // TODO: make this more generic

    // Store the global index information for repairing the remote index information
    std::map<int,SLList<T1> > globalIndices;
    storeGlobalIndicesOfRemoteIndices(globalIndices, oocomm.remoteIndices(), indexSet);
    indexSet.beginResize();

    for(VertexIterator vertex = graph.begin(), vend=graph.end(); vertex != vend; ++vertex) {
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

#if HAVE_PARMETIS
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
    void createSendBuf(std::vector<GI>& ownerVec, std::set<GI>& overlapVec, char *sendBuf, int buffersize, MPI_Comm comm) {
      // Pack owner vertices
      std::size_t s=ownerVec.size();
      int pos=0;
      MPI_Pack(&s, 1, MPITraits<std::size_t>::getType(), sendBuf, buffersize, &pos, comm);
      MPI_Pack(&(ownerVec[0]), s, MPITraits<GI>::getType(), sendBuf, buffersize, &pos, comm);
      s = overlapVec.size();
      MPI_Pack(&s, 1, MPITraits<std::size_t>::getType(), sendBuf, buffersize, &pos, comm);
      typedef typename std::set<GI>::iterator Iter;
      for(Iter i=overlapVec.begin(), end= overlapVec.end(); i != end; ++i)
        MPI_Pack(const_cast<GI*>(&(*i)), 1, MPITraits<GI>::getType(), sendBuf, buffersize, &pos, comm);
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
    void saveRecvBuf(char *recvBuf, int bufferSize, std::vector<GI>& ownerVec, std::set<GI>& overlapVec, MPI_Comm comm) {
      int size;
      int pos=0;
      // unpack owner vertices
      MPI_Unpack(recvBuf, bufferSize, &pos, &size, 1, MPITraits<std::size_t>::getType(), comm);
      int start=ownerVec.size();
      ownerVec.resize(ownerVec.size()+size);
      MPI_Unpack(recvBuf, bufferSize, &pos, &(ownerVec[start]), size, MPITraits<GI>::getType(), comm);

      // unpack overlap vertices
      MPI_Unpack(recvBuf, bufferSize, &pos, &size, 1, MPITraits<std::size_t>::getType(), comm);
      typename std::set<GI>::const_iterator ipos = overlapVec.begin();
      for(; size>0; --size) {
        GI gi;
        MPI_Unpack(recvBuf, bufferSize, &pos, &gi, 1, MPITraits<GI>::getType(), comm);
        ipos=overlapVec.insert(ipos, gi);
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
    void getDomain(const MPI_Comm& comm, int *part, int numOfOwnVtx, int nparts, int *myDomain, int domainMapping[]) {
      int npes, mype;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &mype);
      MPI_Status status;

      *myDomain = -1;
      int i=0;
      int j=0;

      int domain[nparts];
      int assigned[npes];
      // init
      for (i=0; i<nparts; i++) {
        domainMapping[i] = -1;
        domain[i] = 0;
      }
      for (i=0; i<npes; i++) {
        assigned[i] = -0;
      }
      // count the occurance of domains
      for (i=0; i<numOfOwnVtx; i++) {
        domain[part[i]]++;
      }

      int *domainMatrix = new int[npes * nparts];
      // init
      for(i=0; i<npes*nparts; i++) {
        domainMatrix[i]=-1;
      }

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
        maxOccurance = 0;
      }

      delete[] domainMatrix;

    }



    /**
     * @brief Merge the owner/overlap vectors
     *
     * This function merges and adds the vertices of a owner/overlap
     * vector to a result owner/overlap vector
     *
     * @param &ownerVec a global index vector contains the owner vertices to merge/add
     * @param &overlapSet a global index set contains the overlap vertices to merge/add
     */
    template<class GI>
    void mergeVec(std::vector<GI>& ownerVec, std::set<GI>& overlapSet) {

      // Sort the owners
      std::sort(ownerVec.begin(), ownerVec.end());
      typedef typename std::vector<GI>::const_iterator VIter;
#ifdef DEBUG_REPART
      // Safty check for duplicates.
      if(ownerVec.size()>0)
      {
        VIter old=ownerVec.begin();
        for(VIter i=old+1, end=ownerVec.end(); i != end; old=i++)
        {
          if(*i==*old)
            throw "Huch!";
        }
      }

#endif

      typedef typename std::set<GI>::iterator SIter;
      VIter v=ownerVec.begin(), vend=ownerVec.end();
      for(SIter s=overlapSet.begin(), send=overlapSet.end(); s!=send;)
      {
        while(v!=vend && *v<*s) ++v;
        if(v!=vend && *v==*s) {
          // Move to the next element before erasing
          // thus s stays valid!
          SIter tmp=s;
          ++s;
          overlapSet.erase(tmp);
        }else
          ++s;
      }
    }

    class ParmetisDuneIndexMap
    {
    public:
      template<class Graph, class OOComm>
      ParmetisDuneIndexMap(const Graph& graph, const OOComm& com);
      int toParmetis(int i)
      {
        return duneToParmetis[i];
      }
      int toLocalParmetis(int i)
      {
        return duneToParmetis[i]-base_;
      }
      int operator[](int i)
      {
        return duneToParmetis[i];
      }
      int toDune(int i)
      {
        return parmetisToDune[i]-base_;
      }
      int numOfOwnVtx()
      {
        return parmetisToDune.size();
      }
      int* vtxDist()
      {
        return &vtxDist_[0];
      }

    private:
      int base_;
      std::vector<int> duneToParmetis;
      std::vector<int> parmetisToDune;
      // range of vertices for processor i: vtxdist[i] to vtxdist[i+1] (parmetis global)
      std::vector<int> vtxDist_;
    };

    template<class G, class OOComm>
    ParmetisDuneIndexMap::ParmetisDuneIndexMap(const G& graph, const OOComm& oocomm)
      : duneToParmetis(graph.noVertices(), -1), vtxDist_(oocomm.communicator().size()+1)
    {
      int npes=oocomm.communicator().size(), mype=oocomm.communicator().rank();

      typedef typename OOComm::ParallelIndexSet::const_iterator Iterator;
      typedef typename OOComm::OwnerSet OwnerSet;

      int numOfOwnVtx=0;
      Iterator end = oocomm.indexSet().end();
      for(Iterator index = oocomm.indexSet().begin(); index != end; ++index) {
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
      base_=base;

      // The type of the const vertex iterator.
      typedef typename G::ConstVertexIterator VertexIterator;
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
      VertexIterator vend = graph.end();
      for(VertexIterator vertex = graph.begin(); vertex != vend; ++vertex) {
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
    void getNeighbor(const Graph& g, int* part,
                     typename Graph::VertexDescriptor vtx, const IS& indexSet,
                     ParmetisDuneIndexMap& parmetisVtxMapping, int toPe,
                     std::set<GI>& neighbor) {
      typedef typename Graph::ConstEdgeIterator Iter;
      for(Iter edge=g.beginEdges(vtx), end=g.endEdges(vtx); edge!=end; ++edge)
      {
        const typename IS::IndexPair* pindex = indexSet.pair(edge.target());
        assert(pindex);
        if(part[parmetisVtxMapping.toLocalParmetis(pindex->local())]!=toPe)
          // is sent to another process and therefore becomes overlap
          neighbor.insert(pindex->global());
      }
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
    template<class OwnerSet, class G, class IS, class GI>
    void getOwnerOverlapVec(const G& graph, int *part, IS& indexSet, ParmetisDuneIndexMap& parmetisVtxMapping,
                            int myPe, int toPe, std::vector<GI>& ownerVec, std::set<GI>& overlapSet) {

      //typedef typename IndexSet::const_iterator Iterator;
      typedef typename IS::const_iterator Iterator;
      for(Iterator index = indexSet.begin(); index != indexSet.end(); ++index) {
        // Only Process owner vertices, the others are not in the parmetis graph.
        if(OwnerSet::contains(index->local().attribute()))
        {
          if(part[parmetisVtxMapping.toLocalParmetis(index->local())]==toPe)
          {
            getNeighbor<OwnerSet>(graph, part, index->local(), indexSet,
                                  parmetisVtxMapping, toPe, overlapSet);
            ownerVec.push_back(index->global());
          }
        }
      }
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
     * @param data the mapping of local index to ParMETIS global index.
     * @param &xadj the ParMETIS xadj array
     * @param &adjncy the ParMETIS adjncy array
     */
    template<class F, class G, class IS>
    void getAdjArrays(G& graph, IS& indexSet, ParmetisDuneIndexMap& data, int *xadj, int *adjncy)
    {
      int i=0, j=0;

      // The type of the const vertex iterator.
      typedef typename G::ConstVertexIterator VertexIterator;
      //typedef typename IndexSet::const_iterator Iterator;
      typedef typename IS::const_iterator Iterator;

      VertexIterator vend = graph.end();
      Iterator end;

      for(VertexIterator vertex = graph.begin(); vertex != vend; ++vertex) {
        if (isOwner<F>(indexSet,*vertex)) {
          // The type of const edge iterator.
          typedef typename G::ConstEdgeIterator EdgeIterator;
          EdgeIterator eend = vertex.end();
          xadj[j] = i;
          j++;
          for(EdgeIterator edge = vertex.begin(); edge != eend; ++edge) {
            if(*vertex!=edge.source()) {
              // This should never happen as vertex is an iterator positioned
              // at the source of all edge that one gets with VertexIterator::begin()
              throw "Something weired happened!";
            }
            // The type of the edge weights
            typedef typename EdgeIterator::Weight Weight;
            //TODO weights are not considered in this version
            // Get the egde weight
            // const Weight& weight=edge.weight();
            adjncy[i] = data.toParmetis(edge.target());
            i++;
          }
        }
      }
      xadj[j] = i;
      j++;

    }
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
   * @param[out] datari Pointer to store the remote index information
   * for send the data from the original partitioning to the new one in.
   */
  template<class G, class T1, class T2>
  bool graphRepartition(const G& graph, Dune::OwnerOverlapCopyCommunication<T1,T2>& oocomm, int nparts,
                        Dune::OwnerOverlapCopyCommunication<T1,T2>*& outcomm,
                        typename Dune::OwnerOverlapCopyCommunication<T1,T2>::RemoteIndices*& datari)
  {
    MPI_Comm comm=oocomm.communicator();
    oocomm.buildGlobalLookup(graph.noVertices());
    fillIndexSetHoles(graph, oocomm);

    // simple precondition checks

#ifdef PERF_REPART
    // Profiling variables
    double t1=0.0, t2=0.0, t3=0.0, t4=0.0, tSum=0.0;
#endif

    // Common variables
    int i=0, j=0;

    // MPI variables
    int npes = oocomm.communicator().size();
    int mype = oocomm.communicator().rank();

    MPI_Status status;

    assert(nparts<=npes);

    typedef typename  Dune::OwnerOverlapCopyCommunication<T1,T2>::ParallelIndexSet::GlobalIndex GI;
    typedef std::vector<GI> IntVector;
    int myDomain;

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

    // Create the vtxdist array and parmetisVtxMapping.
    // Global communications are necessary
    // The parmetis global identifiers for the owner vertices.
    ParmetisDuneIndexMap indexMap(graph,oocomm);


    // Create the xadj and adjncy arrays
    typedef typename  Dune::OwnerOverlapCopyCommunication<T1,T2> OOComm;
    typedef typename  OOComm::OwnerSet OwnerSet;

    int numOfVtx = oocomm.indexSet().size();
    int *xadj = new  int[indexMap.numOfOwnVtx()+1];
    int *adjncy = new int[graph.noEdges()];
    getAdjArrays<OwnerSet>(graph, oocomm.globalLookup(), indexMap,
                           xadj, adjncy);

    //
    // 2) Call ParMETIS
    //
    //
    idxtype *part = new idxtype[indexMap.numOfOwnVtx()];
    int numflag=0, wgtflag=0, options[10], edgecut=0, ncon=1;
    float *tpwgts = NULL;
    float ubvec[1];
    options[0] = 1; // 0=default, 1=options are defined in [1]+[2]
#ifdef DEBUG_REPART
    options[1] = 3; // show info: 0=no message
#else
    options[1] = 0; // show info: 0=no message
#endif
    options[2] = 1; // random number seed, default is 15
    wgtflag = 0;
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

    //=======================================================
    // ParMETIS_V3_PartKway
    //=======================================================
    ParMETIS_V3_PartKway(indexMap.vtxDist(), xadj, adjncy,
                         NULL, NULL, &wgtflag,
                         &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &const_cast<MPI_Comm&>(comm));


    delete[] xadj;
    delete[] adjncy;

#ifdef DEBUG_REPART
    if (mype == 0) {
      std::cout<<std::endl;
      std::cout<<"ParMETIS_V3_PartKway reported a cut of "<<edgecut<<std::endl;
      std::cout<<std::endl;
    }
    std::cout<<mype<<": PARMETIS-Result: ";
    for(i=0; i < indexMap.vtxDist()[mype+1]-indexMap.vtxDist()[mype]; ++i) {
      std::cout<<part[i]<<" ";
    }
    std::cout<<std::endl;
#endif
#ifdef PERF_REPART
    // stop the time for step 2)
    t2=MPI_Wtime()-t2;
    // reset timer for step 3)
    t3=MPI_Wtime();
#endif


    //
    // 3) Find a optimal domain based on the ParMETIS repatitioning
    //    result
    //

    int domainMapping[nparts];
    getDomain(comm, part, indexMap.numOfOwnVtx(), nparts, &myDomain, domainMapping);
#ifdef DEBUG_REPART
    std::cout<<mype<<": myDomain: "<<myDomain<<std::endl;
    std::cout<<mype<<": DomainMapping: ";
    for(j=0; j<nparts; j++) {
      std::cout<<" do: "<<j<<" pe: "<<domainMapping[j]<<" ";
    }
    std::cout<<std::endl;
#endif
    // translate the domain number to real process number
    int *newPartition = new int[indexMap.numOfOwnVtx()];
    for(j=0; j<indexMap.numOfOwnVtx(); j++)
      newPartition[j] = domainMapping[part[j]];

    // Make a domain mapping for the indexset and translate
    //domain number to real process number
    // domainMapping is the one of parmetis, that is without
    // the overlap/copy vertices
    IntVector setPartition(oocomm.indexSet().size(), -1);

    typedef typename  OOComm::ParallelIndexSet::const_iterator Iterator;
    i=0; // parmetis index
    for(Iterator index = oocomm.indexSet().begin(); index != oocomm.indexSet().end(); ++index)
      if(OwnerSet::contains(index->local().attribute())) {
        setPartition[index->local()]=domainMapping[part[i++]];
      }

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

    int *sendTo = new int[npes];
    int *recvFrom = new int[npes];
    int *buf = new int[npes];
    // init the buffers
    for(j=0; j<npes; j++) {
      sendTo[j] = 0;
      recvFrom[j] = 0;
      buf[j] = 0;
    }

    // the max number of vertices is stored in the sendTo buffer,
    // not the number of vertices to send! Because the max number of Vtx
    // is used as the fixed buffer size by the MPI send/receive calls

    // TODO: optimize buffer size
    bool existentOnNextLevel=false;

    for(i=0; i<indexMap.numOfOwnVtx(); ++i) {
      if (newPartition[i]!=mype) {
        if (sendTo[newPartition[i]]==0) {
          sendTo[newPartition[i]] = numOfVtx;
          buf[newPartition[i]] = numOfVtx;
        }
      }
      else
        existentOnNextLevel=true;
    }

    // The own "send to" array is sent to the next process and so on.
    // Each process receive such a array and pick up the
    // corresponding "receive from" value. This value define the size
    // of the buffer containing the vertices to receive by the next step.
    // TODO: not really a ring communication
    int pe=0;
    int src = (mype-1+npes)%npes;
    int dest = (mype+1)%npes;

    // ring communication, we need n-1 communication for n processors
    for (i=0; i<npes-1; i++) {
      MPI_Sendrecv_replace(buf, npes, MPI_INT, dest, 0, src, 0, comm, &status);
      // pe is the process of the actual received buffer
      pe = ((mype-1-i)+npes)%npes;
      recvFrom[pe] = buf[mype]; // pick up the "recv from" value for myself
      if(recvFrom[pe]>0)
        existentOnNextLevel=true;
    }
    delete[] buf;

#ifdef DEBUG_REPART
    std::cout<<mype<<": recvFrom: ";
    for(i=0; i<npes; i++) {
      std::cout<<recvFrom[i]<<" ";
    }
    std::cout<<std::endl<<std::endl;
    std::cout<<mype<<": sendTo: ";
    for(i=0; i<npes; i++) {
      std::cout<<sendTo[i]<<" ";
    }
    std::cout<<std::endl<<std::endl;
#endif

    //
    // 4.2) Start the communication
    //

    // Get all the owner and overlap vertices for myself ans save
    // it in the vectors myOwnerVec and myOverlapVec.
    // The received vertices from the other processes are simple
    // added to these vector.
    //


    typedef typename OOComm::ParallelIndexSet::GlobalIndex GI;
    IntVector myOwnerVec;
    std::set<GI> myOverlapSet;
    IntVector sendOwnerVec;
    std::set<GI> sendOverlapSet;

    getOwnerOverlapVec<OwnerSet>(graph, newPartition, oocomm.globalLookup(), indexMap,
                                 mype, mype, myOwnerVec, myOverlapSet);

    for(i=0; i < npes; ++i) {
      // the rank of the process defines the sending order,
      // so it starts naturally by 0

      if (i==mype) {
        for(j=0; j < npes; ++j) {
          if (sendTo[j]>0) {
            // clear the vector for sending
            sendOwnerVec.clear();
            sendOverlapSet.clear();
            // get all owner and overlap vertices for process j and save these
            // in the vectors sendOwnerVec and sendOverlapSet
            getOwnerOverlapVec<OwnerSet>(graph, newPartition, oocomm.globalLookup(), indexMap,
                                         mype, j, sendOwnerVec, sendOverlapSet);
            // +2, we need 2 integer more for the length of each part
            // (owner/overlap) of the array
            int buffersize=0;
            int tsize;
            MPI_Pack_size(1, MPI_INT, oocomm.communicator(), &buffersize);
            MPI_Pack_size(1, MPI_INT, oocomm.communicator(), &tsize);
            buffersize +=tsize;
            MPI_Pack_size(sendTo[j], MPITraits<GI>::getType(), oocomm.communicator(), &tsize);
            buffersize += tsize;

            char* sendBuf = new char[buffersize];
#ifdef DEBUG_REPART
            std::cout<<mype<<" sending "<<sendOwnerVec.size()<<" owner and "<<
            sendOverlapSet.size()<<" overlap to "<<j<<" buffersize="<<buffersize<<std::endl;
#endif
            createSendBuf(sendOwnerVec, sendOverlapSet, sendBuf, buffersize, oocomm.communicator());
            MPI_Send(sendBuf, buffersize, MPI_PACKED, j, 0, oocomm.communicator());
            delete[] sendBuf;
          }
        }
      } else { // All the other processes have to wait for receive...
        if (recvFrom[i]>0) {
          // Calculate buffer size
          int buffersize=0;
          int tsize;
          MPI_Pack_size(1, MPI_INT, oocomm.communicator(), &buffersize);
          MPI_Pack_size(1, MPI_INT, oocomm.communicator(), &tsize);
          buffersize +=tsize;
          MPI_Pack_size(recvFrom[i], MPITraits<GI>::getType(), oocomm.communicator(), &tsize);
          buffersize += tsize;
          char* recvBuf = new char[buffersize];
#ifdef DEBUG_REPART
          std::cout<<mype<<" receiving "<<recvFrom[i]<<" from "<<i<<" buffersize="<<buffersize<<std::endl;
#endif
          MPI_Recv(recvBuf, buffersize, MPI_PACKED, i, 0, oocomm.communicator(), &status);
          saveRecvBuf(recvBuf, buffersize, myOwnerVec, myOverlapSet, oocomm.communicator());
          delete[] recvBuf;
        }
      }
    }
    delete[] newPartition;


    // After all the vertices are received, the vectors must
    // be "merged" together to create the final vectors.
    // Because some vertices that are sent as overlap could now
    // already included as owner vertiecs in the new partition
    mergeVec(myOwnerVec, myOverlapSet);

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

    oocomm.communicator().barrier();
    MPI_Comm_split(oocomm.communicator(), color, 0, &outputComm);
    outcomm = new OOComm(outputComm);

    datari = new  typename OOComm::RemoteIndices(oocomm.indexSet(),outcomm->indexSet(), oocomm.communicator());
    typename OOComm::ParallelIndexSet& outputIndexSet = outcomm->indexSet();

    outputIndexSet.beginResize();
    // 1) add the owner vertices
    typedef typename OOComm::ParallelIndexSet::LocalIndex LocalIndex;
    typedef typename IntVector::const_iterator VIter;
    i=0;
    for(VIter g=myOwnerVec.begin(), end =myOwnerVec.end(); g!=end; ++g, ++i ) {
      outputIndexSet.add(*g,LocalIndex(i, OwnerOverlapCopyAttributeSet::owner, true));
    }
    // Trick to free memory
    myOwnerVec.clear();
    myOwnerVec.swap(myOwnerVec);

    // 2) add the overlap vertices
    typedef typename std::set<GI>::const_iterator SIter;
    for(SIter g=myOverlapSet.begin(), end=myOverlapSet.end(); g!=end; ++g, i++) {
      outputIndexSet.add(*g,LocalIndex(i, OwnerOverlapCopyAttributeSet::copy, true));
    }
    myOverlapSet.clear();
    outputIndexSet.endResize();
    outputIndexSet.renumberLocal();

    // build the remoteIndices for the transfer of vertices
    // according to the repartition
    //
    datari->template rebuild<true>();
    if(color != MPI_UNDEFINED) {
      outcomm->remoteIndices().template rebuild<true>();

#ifdef DEBUG_REPART
      if(outcomm->communicator().size()==0) {
        // Check that all indices are owner
        bool correct=true;
        for(Iterator index = outcomm->indexSet().begin(); index != outcomm->indexSet().end(); ++index)
          if(!OwnerSet::contains(index->local().attribute())) {
            std::cout<<*index<<" is overlap!!"<<std::endl;
            correct=false;
          }
        if(!correct)
          throw "hich";
      }
#endif
    }

    // release the memory
    delete[] sendTo;
    delete[] recvFrom;
    delete[] part;

    MPI_Barrier(comm);

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
#endif // HAVE_PARMETIS
#endif // HAVE_MPI
} // end of namespace Dune
#endif
