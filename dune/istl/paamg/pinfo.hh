// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
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
      typedef Communication<void*> MPICommunicator;
      typedef EmptySet<int> CopyFlags;
      typedef AllSet<int> OwnerSet;

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
      void copyOwnerToAll([[maybe_unused]] V& v, [[maybe_unused]] V& v1) const
      {}

      template<class V>
      void project([[maybe_unused]] V& v) const
      {}

      template<class T1, class T2>
      void dot (const T1&, const T1&, T2&) const
      {
        // This function should never be called
        std::abort();
      }

      template<class T1>
      typename FieldTraits<typename T1::field_type>::real_type norm (const T1&) const
      {
        // This function should never be called
        std::abort();
      }

      template<class T>
      SequentialInformation(const Communication<T>&)
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
