// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMGTRANSFER_HH
#define DUNE_AMGTRANSFER_HH

#include <dune/istl/bvector.hh>
#include <dune/istl/matrixredistribute.hh>
#include <dune/istl/paamg/pinfo.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/paamg/aggregates.hh>
#include <dune/common/exceptions.hh>

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
     * @brief Prolongation and restriction for amg.
     */
    template<class V1, class V2, class T>
    class Transfer
    {

    public:
      typedef V1 Vertex;
      typedef V2 Vector;

      template<typename T1, typename R>
      static void prolongateVector(const AggregatesMap<Vertex>& aggregates, Vector& coarse, Vector& fine,
                                   Vector& fineRedist,T1 damp, R& redistributor=R());

      template<typename T1, typename R>
      static void prolongateVector(const AggregatesMap<Vertex>& aggregates, Vector& coarse, Vector& fine,
                                   T1 damp);

      static void restrictVector(const AggregatesMap<Vertex>& aggregates, Vector& coarse, const Vector& fine,
                                 T& comm);
    };

    template<class V,class V1>
    class Transfer<V,V1, SequentialInformation>
    {
    public:
      typedef V Vertex;
      typedef V1 Vector;
      typedef RedistributeInformation<SequentialInformation> Redist;
      template<typename T1>
      static void prolongateVector(const AggregatesMap<Vertex>& aggregates, Vector& coarse, Vector& fine,
                                   Vector& fineRedist, T1 damp,
                                   const SequentialInformation& comm=SequentialInformation(),
                                   const Redist& redist=Redist());
      template<typename T1>
      static void prolongateVector(const AggregatesMap<Vertex>& aggregates, Vector& coarse, Vector& fine,
                                   T1 damp,
                                   const SequentialInformation& comm=SequentialInformation());


      static void restrictVector(const AggregatesMap<Vertex>& aggregates, Vector& coarse, const Vector& fine,
                                 const SequentialInformation& comm);
    };

#if HAVE_MPI

    template<class V,class V1, class T1, class T2>
    class Transfer<V,V1,OwnerOverlapCopyCommunication<T1,T2> >
    {
    public:
      typedef V Vertex;
      typedef V1 Vector;
      typedef RedistributeInformation<OwnerOverlapCopyCommunication<T1,T2> > Redist;
      template<typename T3>
      static void prolongateVector(const AggregatesMap<Vertex>& aggregates, Vector& coarse, Vector& fine,
                                   Vector& fineRedist, T3 damp, OwnerOverlapCopyCommunication<T1,T2>& comm,
                                   const Redist& redist);
      template<typename T3>
      static void prolongateVector(const AggregatesMap<Vertex>& aggregates, Vector& coarse, Vector& fine,
                                   T3 damp, OwnerOverlapCopyCommunication<T1,T2>& comm);

      static void restrictVector(const AggregatesMap<Vertex>& aggregates, Vector& coarse, const Vector& fine,
                                 OwnerOverlapCopyCommunication<T1,T2>& comm);
    };

#endif

    template<class V, class V1>
    template<typename T>
    inline void
    Transfer<V,V1,SequentialInformation>::prolongateVector(const AggregatesMap<Vertex>& aggregates,
                                                           Vector& coarse, Vector& fine,
                                                           [[maybe_unused]] Vector& fineRedist,
                                                           T damp,
                                                           [[maybe_unused]] const SequentialInformation& comm,
                                                           [[maybe_unused]] const Redist& redist)
    {
      prolongateVector(aggregates, coarse, fine, damp);
    }
    template<class V, class V1>
    template<typename T>
    inline void
    Transfer<V,V1,SequentialInformation>::prolongateVector(const AggregatesMap<Vertex>& aggregates,
                                                           Vector& coarse, Vector& fine,
                                                           T damp,
                                                           [[maybe_unused]] const SequentialInformation& comm)
    {
      typedef typename Vector::iterator Iterator;

      Iterator end = coarse.end();
      Iterator begin= coarse.begin();
      for(; begin!=end; ++begin)
        *begin*=damp;
      end=fine.end();
      begin=fine.begin();

      for(Iterator block=begin; block != end; ++block) {
        std::ptrdiff_t index=block-begin;
        const Vertex& vertex = aggregates[index];
        if(vertex != AggregatesMap<Vertex>::ISOLATED)
          *block += coarse[aggregates[index]];
      }
    }

    template<class V, class V1>
    inline void
    Transfer<V,V1,SequentialInformation>::restrictVector(const AggregatesMap<Vertex>& aggregates,
                                                         Vector& coarse,
                                                         const Vector& fine,
                                                         [[maybe_unused]] const SequentialInformation& comm)
    {
      // Set coarse vector to zero
      coarse=0;

      typedef typename Vector::const_iterator Iterator;
      Iterator end = fine.end();
      Iterator begin=fine.begin();

      for(Iterator block=begin; block != end; ++block) {
        const Vertex& vertex = aggregates[block-begin];
        if(vertex != AggregatesMap<Vertex>::ISOLATED)
          coarse[vertex] += *block;
      }
    }

#if HAVE_MPI
    template<class V, class V1, class T1, class T2>
    template<typename T3>
    inline void Transfer<V,V1,OwnerOverlapCopyCommunication<T1,T2> >::prolongateVector(const AggregatesMap<Vertex>& aggregates,
                                                                                       Vector& coarse, Vector& fine,
                                                                                       Vector& fineRedist, T3 damp,
                                                                                       OwnerOverlapCopyCommunication<T1,T2>& comm,
                                                                                       const Redist& redist)
    {
      if(fineRedist.size()>0)
        // we operated on the coarse level
        Transfer<V,V1,SequentialInformation>::prolongateVector(aggregates, coarse, fineRedist, damp);

      // TODO This could be accomplished with one communication, too!
      redist.redistributeBackward(fine, fineRedist);
      comm.copyOwnerToAll(fine,fine);
    }

    template<class V, class V1, class T1, class T2>
    template<typename T3>
    inline void Transfer<V,V1,OwnerOverlapCopyCommunication<T1,T2> >::prolongateVector(
      const AggregatesMap<Vertex>& aggregates,
      Vector& coarse, Vector& fine, T3 damp,
      [[maybe_unused]] OwnerOverlapCopyCommunication<T1,T2>& comm)
    {
      Transfer<V,V1,SequentialInformation>::prolongateVector(aggregates, coarse, fine, damp);
    }
    template<class V, class V1, class T1, class T2>
    inline void Transfer<V,V1,OwnerOverlapCopyCommunication<T1,T2> >::restrictVector(const AggregatesMap<Vertex>& aggregates,
                                                                                     Vector& coarse, const Vector& fine,
                                                                                     OwnerOverlapCopyCommunication<T1,T2>& comm)
    {
      Transfer<V,V1,SequentialInformation>::restrictVector(aggregates, coarse, fine, SequentialInformation());
      // We need this here to avoid it in the smoothers on the coarse level.
      // There (in the preconditioner d is const.
      comm.project(coarse);
    }
#endif
    /** @} */
  }    // namspace Amg
}     // namspace Dune
#endif
