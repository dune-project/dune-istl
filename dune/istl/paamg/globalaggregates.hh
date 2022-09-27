// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GLOBALAGGREGATES_HH
#define DUNE_GLOBALAGGREGATES_HH

/**
 * @addtogroup ISTL_PAAMG
 *
 * @{
 */
/** @file
 * @author Markus Blatt
 * @brief Provdes class for identifying aggregates globally.
 */

#include "aggregates.hh"
#include "pinfo.hh"
#include <dune/common/parallel/indexset.hh>

namespace Dune
{
  namespace Amg
  {

    template<typename T, typename TI>
    struct GlobalAggregatesMap
    {
    public:
      typedef TI ParallelIndexSet;

      typedef typename ParallelIndexSet::GlobalIndex GlobalIndex;

      typedef typename ParallelIndexSet::GlobalIndex IndexedType;

      typedef typename ParallelIndexSet::LocalIndex LocalIndex;

      typedef T Vertex;

      GlobalAggregatesMap(AggregatesMap<Vertex>& aggregates,
                          const GlobalLookupIndexSet<ParallelIndexSet>& indexset)
        : aggregates_(aggregates), indexset_(indexset)
      {}

      inline const GlobalIndex& operator[](std::size_t index) const
      {
        const Vertex& aggregate = aggregates_[index];
        if(aggregate >= AggregatesMap<Vertex>::ISOLATED) {
          assert(aggregate != AggregatesMap<Vertex>::UNAGGREGATED);
          return isolatedMarker;
        }else{
          const Dune::IndexPair<GlobalIndex,LocalIndex >* pair = indexset_.pair(aggregate);
          assert(pair!=0);
          return pair->global();
        }
      }


      inline GlobalIndex& get(std::size_t index)
      {
        const Vertex& aggregate = aggregates_[index];
        assert(aggregate < AggregatesMap<Vertex>::ISOLATED);
        const Dune::IndexPair<GlobalIndex,LocalIndex >* pair = indexset_.pair(aggregate);
        assert(pair!=0);
        return const_cast<GlobalIndex&>(pair->global());
      }

      class Proxy
      {
      public:
        Proxy(const GlobalLookupIndexSet<ParallelIndexSet>& indexset, Vertex& aggregate)
          : indexset_(&indexset), aggregate_(&aggregate)
        {}

        Proxy& operator=(const GlobalIndex& global)
        {
          if(global==isolatedMarker)
            *aggregate_ = AggregatesMap<Vertex>::ISOLATED;
          else{
            //assert(global < AggregatesMap<Vertex>::ISOLATED);
            *aggregate_ = indexset_->operator[](global).local();
          }
          return *this;
        }
      private:
        const GlobalLookupIndexSet<ParallelIndexSet>* indexset_;
        Vertex* aggregate_;
      };

      inline Proxy operator[](std::size_t index)
      {
        return Proxy(indexset_, aggregates_[index]);
      }

      inline void put(const GlobalIndex& global, size_t i)
      {
        aggregates_[i]=indexset_[global].local();

      }

    private:
      AggregatesMap<Vertex>& aggregates_;
      const GlobalLookupIndexSet<ParallelIndexSet>& indexset_;
      static const GlobalIndex isolatedMarker;
    };

    template<typename T, typename TI>
    const typename TI::GlobalIndex GlobalAggregatesMap<T,TI>::isolatedMarker =
      std::numeric_limits<typename TI::GlobalIndex>::max();

    template<typename T, typename TI>
    struct AggregatesGatherScatter
    {
      typedef TI ParallelIndexSet;
      typedef typename ParallelIndexSet::GlobalIndex GlobalIndex;

      static const GlobalIndex& gather(const GlobalAggregatesMap<T,TI>& ga, size_t i)
      {
        return ga[i];
      }

      static void scatter(GlobalAggregatesMap<T,TI>& ga, GlobalIndex global, size_t i)
      {
        ga[i]=global;
      }
    };

    template<typename T, typename O, typename I>
    struct AggregatesPublisher
    {};

#if HAVE_MPI

#endif

  } // namespace Amg

#if HAVE_MPI
  // forward declaration
  template<class T1, class T2>
  class OwnerOverlapCopyCommunication;
#endif

  namespace Amg
  {

#if HAVE_MPI
    /**
     * @brief Utility class for publishing the aggregate number
     * of the DOFs in the overlap to other processors and convert
     * them to local indices.
     * @tparam T The type of the vertices
     * @tparam O The set of overlap flags.
     * @tparam T1 The type of the global indices.
     * @tparam T2 The type of the local indices.
     */
    template<typename T, typename O, typename T1, typename T2>
    struct AggregatesPublisher<T,O,OwnerOverlapCopyCommunication<T1,T2> >
    {
      typedef T Vertex;
      typedef O OverlapFlags;
      typedef OwnerOverlapCopyCommunication<T1,T2> ParallelInformation;
      typedef typename ParallelInformation::GlobalLookupIndexSet GlobalLookupIndexSet;
      typedef typename ParallelInformation::ParallelIndexSet IndexSet;

      static void publish(AggregatesMap<Vertex>& aggregates,
                          ParallelInformation& pinfo,
                          const GlobalLookupIndexSet& globalLookup)
      {
        typedef Dune::Amg::GlobalAggregatesMap<Vertex,IndexSet> GlobalMap;
        GlobalMap gmap(aggregates, globalLookup);
        pinfo.copyOwnerToAll(gmap,gmap);
        // communication only needed for ALU
        // (ghosts with same global id as owners on the same process)
        if (SolverCategory::category(pinfo) == static_cast<int>(SolverCategory::nonoverlapping))
          pinfo.copyCopyToAll(gmap,gmap);

        typedef typename ParallelInformation::RemoteIndices::const_iterator Lists;
        Lists lists = pinfo.remoteIndices().find(pinfo.communicator().rank());
        if(lists!=pinfo.remoteIndices().end()) {

          // For periodic boundary conditions we must renumber
          // the aggregates of vertices in the overlap whose owners are
          // on the same process
          Vertex maxAggregate =0;
          typedef typename AggregatesMap<Vertex>::const_iterator Iter;
          for(Iter i=aggregates.begin(), end=aggregates.end(); i!=end; ++i)
            maxAggregate = std::max(maxAggregate, *i);

          // Compute new mapping of aggregates in the overlap that we also own
          std::map<Vertex,Vertex> newMapping;

          // insert all elements into map
          typedef typename ParallelInformation::RemoteIndices::RemoteIndexList
          ::const_iterator RIter;
          for(RIter ri=lists->second.first->begin(), rend = lists->second.first->end();
              ri!=rend; ++ri)
            if(O::contains(ri->localIndexPair().local().attribute()))
              newMapping.insert(std::make_pair(aggregates[ri->localIndexPair().local()],
                                               maxAggregate));
          // renumber
          typedef typename std::map<Vertex,Vertex>::iterator MIter;
          for(MIter mi=newMapping.begin(), mend=newMapping.end();
              mi != mend; ++mi)
            mi->second=++maxAggregate;


          for(RIter ri=lists->second.first->begin(), rend = lists->second.first->end();
              ri!=rend; ++ri)
            if(O::contains(ri->localIndexPair().local().attribute()))
              aggregates[ri->localIndexPair().local()] =
                newMapping[aggregates[ri->localIndexPair().local()]];
        }
      }
    };
#endif

    template<typename T, typename O>
    struct AggregatesPublisher<T,O,SequentialInformation>
    {
      typedef T Vertex;
      typedef SequentialInformation ParallelInformation;
      typedef typename ParallelInformation::GlobalLookupIndexSet GlobalLookupIndexSet;

      static void publish([[maybe_unused]] AggregatesMap<Vertex>& aggregates,
                          [[maybe_unused]] ParallelInformation& pinfo,
                          [[maybe_unused]] const GlobalLookupIndexSet& globalLookup)
      {}
    };

  } // end Amg namespace


#if HAVE_MPI
  template<typename T, typename TI>
  struct CommPolicy<Amg::GlobalAggregatesMap<T,TI> >
  {
    typedef Amg::AggregatesMap<T> Type;
    typedef typename Amg::GlobalAggregatesMap<T,TI>::IndexedType IndexedType;
    typedef SizeOne IndexedTypeFlag;
    static int getSize(const Type&, int)
    {
      return 1;
    }
  };
#endif

} // end Dune namespace
  /* @} */
#endif
