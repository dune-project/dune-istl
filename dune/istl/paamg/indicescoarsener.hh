// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMG_INDICESCOARSENER_HH
#define DUNE_AMG_INDICESCOARSENER_HH

#include <dune/common/parallel/indicessyncer.hh>
#include <vector>
#include "renumberer.hh"

#if HAVE_MPI
#include <dune/istl/owneroverlapcopy.hh>
#endif

#include "pinfo.hh"

namespace Dune
{
  namespace Amg
  {

    /**
     * @addtogroup ISTL_PAAMG
     *
     * @{
     */
    /** @file
     * @author Markus Blatt
     * @brief Provides a class for building the index set
     * and remote indices on the coarse level.
     */

    template<typename T, typename E>
    class IndicesCoarsener
    {};


#if HAVE_MPI

    template<typename T, typename E>
    class ParallelIndicesCoarsener
    {
    public:
      /**
       * @brief The set of excluded attributes
       */
      typedef E ExcludedAttributes;

      /**
       * @brief The type of the parallel information.
       */
      typedef T ParallelInformation;

      typedef typename ParallelInformation::ParallelIndexSet ParallelIndexSet;

      /**
       * @brief The type of the global index.
       */
      typedef typename ParallelIndexSet::GlobalIndex GlobalIndex;

      /**
       * @brief The type of the local index.
       */
      typedef typename ParallelIndexSet::LocalIndex LocalIndex;

      /**
       * @brief The type of the attribute.
       */
      typedef typename LocalIndex::Attribute Attribute;

      /**
       * @brief The type of the remote indices.
       */
      typedef Dune::RemoteIndices<ParallelIndexSet> RemoteIndices;

      /**
       * @brief Build the coarse index set after the aggregatio.
       *
       * @param fineInfo The parallel information at the fine level.
       * @param fineGraph The graph of the fine lecel,
       * @param visitedMap Map for marking vertices as visited.
       * @param aggregates The mapping of unknowns onto aggregates.
       * @param coarseInfo The information about the parallel data decomposition
       * on the coarse level.
       * @return The number of unknowns on the coarse level.
       */
      template<typename Graph, typename VM>
      static typename Graph::VertexDescriptor
      coarsen(ParallelInformation& fineInfo,
              Graph& fineGraph,
              VM& visitedMap,
              AggregatesMap<typename Graph::VertexDescriptor>& aggregates,
              ParallelInformation& coarseInfo,
              typename Graph::VertexDescriptor noAggregates);

    private:
      template<typename G, typename I>
      class ParallelAggregateRenumberer : public AggregateRenumberer<G>
      {
        typedef typename G::VertexDescriptor Vertex;

        typedef I GlobalLookupIndexSet;

        typedef typename GlobalLookupIndexSet::IndexPair IndexPair;

        typedef typename IndexPair::GlobalIndex GlobalIndex;

      public:
        ParallelAggregateRenumberer(AggregatesMap<Vertex>& aggregates, const I& lookup)
          :  AggregateRenumberer<G>(aggregates),  isPublic_(false), lookup_(lookup),
            globalIndex_(std::numeric_limits<GlobalIndex>::max())
        {}


        void operator()(const typename G::ConstEdgeIterator& edge)
        {
          AggregateRenumberer<G>::operator()(edge);
          const IndexPair* pair= lookup_.pair(edge.target());
          if(pair!=0) {
            globalIndex(pair->global());
            attribute(pair->local().attribute());
            isPublic(pair->local().isPublic());
          }
        }

        Vertex operator()([[maybe_unused]] const GlobalIndex& global)
        {
          Vertex current = this->number_;
          this->operator++();
          return current;
        }

        bool isPublic()
        {
          return isPublic_;
        }

        void isPublic(bool b)
        {
          isPublic_ = isPublic_ || b;
        }

        void reset()
        {
          globalIndex_ = std::numeric_limits<GlobalIndex>::max();
          isPublic_=false;
        }

        void attribute(const Attribute& attribute)
        {
          attribute_=attribute;
        }

        Attribute attribute()
        {
          return attribute_;
        }

        const GlobalIndex& globalIndex() const
        {
          return globalIndex_;
        }

        void globalIndex(const GlobalIndex& global)
        {
          globalIndex_ = global;
        }

      private:
        bool isPublic_;
        Attribute attribute_;
        const GlobalLookupIndexSet& lookup_;
        GlobalIndex globalIndex_;
      };

      template<typename Graph, typename VM, typename I>
      static void buildCoarseIndexSet(const ParallelInformation& pinfo,
                                      Graph& fineGraph,
                                      VM& visitedMap,
                                      AggregatesMap<typename Graph::VertexDescriptor>& aggregates,
                                      ParallelIndexSet& coarseIndices,
                                      ParallelAggregateRenumberer<Graph,I>& renumberer);

      template<typename Graph,typename I>
      static void buildCoarseRemoteIndices(const RemoteIndices& fineRemote,
                                           const AggregatesMap<typename Graph::VertexDescriptor>& aggregates,
                                           ParallelIndexSet& coarseIndices,
                                           RemoteIndices& coarseRemote,
                                           ParallelAggregateRenumberer<Graph,I>& renumberer);

    };

    /**
     * @brief Coarsen Indices in the parallel case.
     */
    template<typename G, typename L, typename E>
    class IndicesCoarsener<OwnerOverlapCopyCommunication<G,L>,E>
      : public ParallelIndicesCoarsener<OwnerOverlapCopyCommunication<G,L>,E>
    {};


#endif

    /**
     * @brief Coarsen Indices in the sequential case.
     *
     * Nothing to be coarsened here. Just renumber the aggregates
     * consecutively
     */
    template<typename E>
    class IndicesCoarsener<SequentialInformation,E>
    {
    public:
      template<typename Graph, typename VM>
      static typename Graph::VertexDescriptor
      coarsen(const SequentialInformation & fineInfo,
              Graph& fineGraph,
              VM& visitedMap,
              AggregatesMap<typename Graph::VertexDescriptor>& aggregates,
              SequentialInformation& coarseInfo,
              typename Graph::VertexDescriptor noAggregates);
    };

#if HAVE_MPI
    template<typename T, typename E>
    template<typename Graph, typename VM>
    inline typename Graph::VertexDescriptor
    ParallelIndicesCoarsener<T,E>::coarsen(ParallelInformation& fineInfo,
                                           Graph& fineGraph,
                                           VM& visitedMap,
                                           AggregatesMap<typename Graph::VertexDescriptor>& aggregates,
                                           ParallelInformation& coarseInfo,
                                           [[maybe_unused]] typename Graph::VertexDescriptor noAggregates)
    {
      ParallelAggregateRenumberer<Graph,typename ParallelInformation::GlobalLookupIndexSet> renumberer(aggregates, fineInfo.globalLookup());
      buildCoarseIndexSet(fineInfo, fineGraph, visitedMap, aggregates,
                          coarseInfo.indexSet(), renumberer);
      buildCoarseRemoteIndices(fineInfo.remoteIndices(), aggregates, coarseInfo.indexSet(),
                               coarseInfo.remoteIndices(), renumberer);

      return renumberer;
    }

    template<typename T, typename E>
    template<typename Graph, typename VM, typename I>
    void ParallelIndicesCoarsener<T,E>::buildCoarseIndexSet(const ParallelInformation& pinfo,
                                                            Graph& fineGraph,
                                                            VM& visitedMap,
                                                            AggregatesMap<typename Graph::VertexDescriptor>& aggregates,
                                                            ParallelIndexSet& coarseIndices,
                                                            ParallelAggregateRenumberer<Graph,I>& renumberer)
    {
      // fineGraph is the local subgraph corresponding to the vertices the process owns.
      // i.e. no overlap/copy vertices can be visited traversing the graph
      typedef typename Graph::ConstVertexIterator Iterator;
      typedef typename ParallelInformation::GlobalLookupIndexSet GlobalLookupIndexSet;

      Iterator end = fineGraph.end();
      const GlobalLookupIndexSet& lookup = pinfo.globalLookup();

      coarseIndices.beginResize();

      // Setup the coarse index set and renumber the aggregate consecutively
      // ascending from zero according to the minimum global index belonging
      // to the aggregate
      for(Iterator index = fineGraph.begin(); index != end; ++index) {
        if(aggregates[*index]!=AggregatesMap<typename Graph::VertexDescriptor>::ISOLATED)
          // Isolated vertices will not be represented on the next level.
          // These should only be there if skipIsolated is activiated in
          // the coarsening criterion as otherwise they will be aggregated
          // and should have real aggregate number in the map right now.
          if(!get(visitedMap, *index)) {
            // This vertex was not visited by breadthFirstSearch yet.
            typedef typename GlobalLookupIndexSet::IndexPair IndexPair;
            const IndexPair* pair= lookup.pair(*index);

            renumberer.reset(); // reset attribute and global index.
            if(pair!=0) {
              // vertex is in the index set. Note that not all vertices have
              // to be in the index set, just the ones where communication
              // will happen.
              assert(!ExcludedAttributes::contains(pair->local().attribute()));
              renumberer.attribute(pair->local().attribute());
              renumberer.isPublic(pair->local().isPublic());
              renumberer.globalIndex(pair->global());
            }

            // Reconstruct aggregate and mark vertices as visited
            aggregates.template breadthFirstSearch<false>(*index, aggregates[*index],
                                                          fineGraph, renumberer, visitedMap);

            if(renumberer.globalIndex()!=std::numeric_limits<GlobalIndex>::max()) {
              // vertex is in the index set.
              //std::cout <<" Adding global="<< renumberer.globalIndex()<<" local="<<static_cast<std::size_t>(renumberer)<<std::endl;
              coarseIndices.add(renumberer.globalIndex(),
                                LocalIndex(renumberer, renumberer.attribute(),
                                           renumberer.isPublic()));
            }

            aggregates[*index] = renumberer;
            ++renumberer;
          }
      }

      coarseIndices.endResize();

      assert(static_cast<std::size_t>(renumberer) >= coarseIndices.size());

      // Reset the visited flags
      for(Iterator vertex=fineGraph.begin(); vertex != end; ++vertex)
        put(visitedMap, *vertex, false);
    }

    template<typename T, typename E>
    template<typename Graph, typename I>
    void ParallelIndicesCoarsener<T,E>::buildCoarseRemoteIndices(const RemoteIndices& fineRemote,
                                                                 const AggregatesMap<typename Graph::VertexDescriptor>& aggregates,
                                                                 ParallelIndexSet& coarseIndices,
                                                                 RemoteIndices& coarseRemote,
                                                                 ParallelAggregateRenumberer<Graph,I>& renumberer)
    {
      std::vector<char> attributes(static_cast<std::size_t>(renumberer));

      GlobalLookupIndexSet<ParallelIndexSet> coarseLookup(coarseIndices, static_cast<std::size_t>(renumberer));

      typedef typename RemoteIndices::const_iterator Iterator;
      Iterator end = fineRemote.end();

      for(Iterator neighbour = fineRemote.begin();
          neighbour != end; ++neighbour) {
        int process = neighbour->first;

        assert(neighbour->second.first==neighbour->second.second);

        // Mark all as not known
        typedef typename std::vector<char>::iterator CIterator;

        for(CIterator iter=attributes.begin(); iter!= attributes.end(); ++iter)
          *iter = std::numeric_limits<char>::max();

        auto riEnd = neighbour->second.second->end();

        for(auto index = neighbour->second.second->begin();
            index != riEnd; ++index) {
          if(!E::contains(index->localIndexPair().local().attribute()) &&
             aggregates[index->localIndexPair().local()] !=
             AggregatesMap<typename Graph::VertexDescriptor>::ISOLATED)
          {
            assert(aggregates[index->localIndexPair().local()]<attributes.size());
            if (attributes[aggregates[index->localIndexPair().local()]] != 3)
              attributes[aggregates[index->localIndexPair().local()]] = index->attribute();
          }
        }

        // Build remote index list
        typedef RemoteIndexListModifier<ParallelIndexSet,typename RemoteIndices::Allocator,false> Modifier;
        typedef typename RemoteIndices::RemoteIndex RemoteIndex;
        typedef typename ParallelIndexSet::const_iterator IndexIterator;

        Modifier coarseList = coarseRemote.template getModifier<false,true>(process);

        IndexIterator iend = coarseIndices.end();
        for(IndexIterator index = coarseIndices.begin(); index != iend; ++index)
          if(attributes[index->local()] != std::numeric_limits<char>::max()) {
            // remote index is present
            coarseList.insert(RemoteIndex(Attribute(attributes[index->local()]), &(*index)));
          }
        //std::cout<<coarseRemote<<std::endl;
      }

      // The number of neighbours should not change!
      assert(coarseRemote.neighbours()==fineRemote.neighbours());

      // snyc the index set and the remote indices to recompute missing
      // indices
      IndicesSyncer<ParallelIndexSet> syncer(coarseIndices, coarseRemote);
      syncer.sync(renumberer);

    }

#endif

    template<typename E>
    template<typename Graph, typename VM>
    typename Graph::VertexDescriptor
    IndicesCoarsener<SequentialInformation,E>::coarsen(
      [[maybe_unused]] const SequentialInformation& fineInfo,
      [[maybe_unused]] Graph& fineGraph,
      [[maybe_unused]] VM& visitedMap,
      [[maybe_unused]] AggregatesMap<typename Graph::VertexDescriptor>& aggregates,
      [[maybe_unused]] SequentialInformation& coarseInfo,
      [[maybe_unused]] typename Graph::VertexDescriptor noAggregates)
    {
      return noAggregates;
    }

  } //namespace Amg
} // namespace Dune
#endif
