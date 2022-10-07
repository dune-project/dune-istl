// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMGCONSTRUCTION_HH
#define DUNE_AMGCONSTRUCTION_HH

#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/solvercategory.hh>
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
     * @brief Helper classes for the construction of classes without
     * empty constructor.
     */
    /**
     * @brief Traits class for generically constructing non default
     * constructable types.
     *
     * Needed because BCRSMatrix and Vector do a deep copy which is
     * too expensive.
     */
    template<typename T>
    struct ConstructionTraits
    {
      /**
       * @brief A type holding all the arguments needed to call the
       * constructor.
       */
      typedef const void* Arguments;

      /**
       * @brief Construct an object with the specified arguments.
       *
       * In the default implementation the copy constructor is called.
       * @param args The arguments for the construction.
       */
      static inline std::shared_ptr<T> construct(Arguments&  args)
      {
        return std::make_shared<T>();
      }
    };

    template<class T, class A>
    struct ConstructionTraits<BlockVector<T,A> >
    {
      typedef const int Arguments;
      static inline std::shared_ptr<BlockVector<T,A>> construct(Arguments& n)
      {
        return std::make_shared<BlockVector<T,A>>(n);
      }
    };

    template<class M, class C>
    struct ParallelOperatorArgs
    {
      ParallelOperatorArgs(std::shared_ptr<M> matrix, const C& comm)
        : matrix_(matrix), comm_(comm)
      {}

      std::shared_ptr<M> matrix_;
      const C& comm_;
    };

#if HAVE_MPI
    struct OwnerOverlapCopyCommunicationArgs
    {
      OwnerOverlapCopyCommunicationArgs(MPI_Comm comm, SolverCategory::Category cat)
        : comm_(comm), cat_(cat)
      {}

      MPI_Comm comm_;
      SolverCategory::Category cat_;
    };
#endif

    struct SequentialCommunicationArgs
    {
      SequentialCommunicationArgs(Communication<void*> comm, [[maybe_unused]] int cat)
        : comm_(comm)
      {}

      Communication<void*> comm_;
    };

  } // end Amg namspace

  // forward declaration
  template<class M, class X, class Y, class C>
  class OverlappingSchwarzOperator;

  template<class M, class X, class Y, class C>
  class NonoverlappingSchwarzOperator;

  namespace Amg
  {
    template<class M, class X, class Y, class C>
    struct ConstructionTraits<OverlappingSchwarzOperator<M,X,Y,C> >
    {
      typedef ParallelOperatorArgs<M,C> Arguments;

      static inline std::shared_ptr<OverlappingSchwarzOperator<M,X,Y,C>> construct(const Arguments& args)
      {
        return std::make_shared<OverlappingSchwarzOperator<M,X,Y,C>>
          (args.matrix_, args.comm_);
      }
    };

    template<class M, class X, class Y, class C>
    struct ConstructionTraits<NonoverlappingSchwarzOperator<M,X,Y,C> >
    {
      typedef ParallelOperatorArgs<M,C> Arguments;

      static inline std::shared_ptr<NonoverlappingSchwarzOperator<M,X,Y,C>> construct(const Arguments& args)
      {
        return std::make_shared<NonoverlappingSchwarzOperator<M,X,Y,C>>
          (args.matrix_, args.comm_);
      }
    };

    template<class M, class X, class Y>
    struct MatrixAdapterArgs
    {
      MatrixAdapterArgs(std::shared_ptr<M> matrix, const SequentialInformation)
        : matrix_(matrix)
      {}

      std::shared_ptr<M> matrix_;
    };

    template<class M, class X, class Y>
    struct ConstructionTraits<MatrixAdapter<M,X,Y> >
    {
      typedef const MatrixAdapterArgs<M,X,Y> Arguments;

      static inline std::shared_ptr<MatrixAdapter<M,X,Y>> construct(Arguments& args)
      {
        return std::make_shared<MatrixAdapter<M,X,Y>>(args.matrix_);
      }
    };

    template<>
    struct ConstructionTraits<SequentialInformation>
    {
      typedef const SequentialCommunicationArgs Arguments;
      static inline std::shared_ptr<SequentialInformation> construct(Arguments& args)
      {
        return std::make_shared<SequentialInformation>(args.comm_);
      }
    };


#if HAVE_MPI

    template<class T1, class T2>
    struct ConstructionTraits<OwnerOverlapCopyCommunication<T1,T2> >
    {
      typedef const OwnerOverlapCopyCommunicationArgs Arguments;

      static inline std::shared_ptr<OwnerOverlapCopyCommunication<T1,T2>> construct(Arguments& args)
      {
        return std::make_shared<OwnerOverlapCopyCommunication<T1,T2>>(args.comm_, args.cat_);
      }
    };

#endif

    /** @} */
  } // namespace Amg
} // namespace Dune
#endif
