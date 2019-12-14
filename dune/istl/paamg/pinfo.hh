// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMG_PINFO_HH
#define DUNE_AMG_PINFO_HH

#include <dune/common/parallel/communication.hh>
#include <dune/common/enumset.hh>

#if HAVE_MPI

#include <dune/common/parallel/mpicommunication.hh>
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

      SolverCategory::Category
      DUNE_DEPRECATED_MSG("use category()")
      getSolverCategory () const {
        return SolverCategory::sequential;
      }

      SolverCategory::Category category () const {
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

      template<class T1, class T2>
      void dot (const T1& x, const T1& y, T2& result) const
      {
        // This function should never be called
        std::abort();
      }

      template<class T1>
      typename FieldTraits<typename T1::field_type>::real_type norm (const T1& x) const
      {
        // This function should never be called
        std::abort();
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
