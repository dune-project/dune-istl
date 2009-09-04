// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_AMG_PINFO_HH
#define DUNE_AMG_PINFO_HH

#include <dune/common/collectivecommunication.hh>
#include <dune/common/enumset.hh>

#if HAVE_MPI

#include <dune/common/mpicollectivecommunication.hh>
#include <dune/istl/mpitraits.hh>
#include <dune/istl/remoteindices.hh>
#include <dune/istl/interface.hh>
#include <dune/istl/communicator.hh>

#endif

#include <dune/istl/solvercategory.hh>
namespace Dune
{
  namespace Amg
  {

    class SequentialInformation
    {
    public:
      typedef CollectiveCommunication<void*> MPICommunicator;
      typedef EmptySet<int> CopyFlags;

      enum {
        category = SolverCategory::sequential
      };

      MPICommunicator communicator() const
      {
        return comm_;
      }

      int procs() const
      {
        return 1;
      }

      template<typename T>
      T globalSum(const T& t) const
      {
        return t;
      }

      typedef int GlobalLookupIndexSet;

      void buildGlobalLookup(std::size_t){};

      void freeGlobalLookup(){};

      const GlobalLookupIndexSet& globalLookup() const
      {
        return gli;
      }

      template<class V>
      void copyOwnerToAll(V& v, V& v1) const
      {}

      template<class V>
      void project(V& v) const
      {}

      SequentialInformation(const CollectiveCommunication<void*>&)
      {}

      SequentialInformation()
      {}

      SequentialInformation(const SequentialInformation&)
      {}
    private:
      MPICommunicator comm_;
      GlobalLookupIndexSet gli;
    };


  } // namespace Amg
} //namespace Dune
#endif
