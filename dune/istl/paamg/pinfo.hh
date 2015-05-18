// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMG_PINFO_HH
#define DUNE_AMG_PINFO_HH

#include <dune/common/parallel/collectivecommunication.hh>
#include <dune/common/enumset.hh>

#if HAVE_MPI

#include <dune/common/parallel/mpicollectivecommunication.hh>
#include <dune/common/parallel/mpitraits.hh>
#include <dune/common/parallel/remoteindices.hh>
#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/communicator.hh>

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
      typedef AllSet<int> OwnerSet;

      enum {
        category = SolverCategory::sequential
      };

      SolverCategory::Category getSolverCategory () const {
        return SolverCategory::sequential;
      }

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

      void buildGlobalLookup(std::size_t){}

      void freeGlobalLookup(){}

      const GlobalLookupIndexSet& globalLookup() const
      {
        return gli;
      }

      template<class V>
      void copyOwnerToAll(V& v, V& v1) const
      {
        DUNE_UNUSED_PARAMETER(v);
        DUNE_UNUSED_PARAMETER(v1);
      }

      template<class V>
      void project(V& v) const
      {
        DUNE_UNUSED_PARAMETER(v);
      }

      template<class T>
      SequentialInformation(const CollectiveCommunication<T>&)
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
