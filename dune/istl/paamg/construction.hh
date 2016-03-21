// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMGCONSTRUCTION_HH
#define DUNE_AMGCONSTRUCTION_HH

#include <dune/common/unused.hh>
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
    class ConstructionTraits
    {
    public:
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
      static inline T* construct(Arguments&  args)
      {
        return new T();
      }

      /**
       * @brief Destroys an object.
       * @param t Pointer to the object to destroy.
       */
      static inline void deconstruct(T* t)
      {
        delete t;
      }

    };

    template<class T, class A>
    class ConstructionTraits<BlockVector<T,A> >
    {
    public:
      typedef const int Arguments;
      static inline BlockVector<T,A>* construct(Arguments& n)
      {
        return new BlockVector<T,A>(n);
      }

      static inline void deconstruct(BlockVector<T,A>* t)
      {
        delete t;
      }
    };

    template<class M, class C>
    struct OverlappingSchwarzOperatorArgs
    {
      OverlappingSchwarzOperatorArgs(M& matrix, C& comm)
        : matrix_(&matrix), comm_(&comm)
      {}

      M* matrix_;
      C* comm_;
    };

    template<class M, class C>
    struct NonoverlappingOperatorArgs
    {
      NonoverlappingOperatorArgs(M& matrix, C& comm)
        : matrix_(&matrix), comm_(&comm)
      {}

      M* matrix_;
      C* comm_;
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
      SequentialCommunicationArgs(CollectiveCommunication<void*> comm, int cat)
        : comm_(comm)
      {
        DUNE_UNUSED_PARAMETER(cat);
      }

      CollectiveCommunication<void*> comm_;
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
    class ConstructionTraits<OverlappingSchwarzOperator<M,X,Y,C> >
    {
    public:
      typedef OverlappingSchwarzOperatorArgs<M,C> Arguments;

      static inline OverlappingSchwarzOperator<M,X,Y,C>* construct(const Arguments& args)
      {
        return new OverlappingSchwarzOperator<M,X,Y,C>(*args.matrix_, *args.comm_);
      }

      static inline void deconstruct(OverlappingSchwarzOperator<M,X,Y,C>* t)
      {
        delete t;
      }
    };

    template<class M, class X, class Y, class C>
    class ConstructionTraits<NonoverlappingSchwarzOperator<M,X,Y,C> >
    {
    public:
      typedef NonoverlappingOperatorArgs<M,C> Arguments;

      static inline NonoverlappingSchwarzOperator<M,X,Y,C>* construct(const Arguments& args)
      {
        return new NonoverlappingSchwarzOperator<M,X,Y,C>(*args.matrix_, *args.comm_);
      }

      static inline void deconstruct(NonoverlappingSchwarzOperator<M,X,Y,C>* t)
      {
        delete t;
      }
    };

    template<class M, class X, class Y>
    struct MatrixAdapterArgs
    {
      MatrixAdapterArgs(M& matrix, const SequentialInformation&)
        : matrix_(&matrix)
      {}

      M* matrix_;
    };

    template<class M, class X, class Y>
    class ConstructionTraits<MatrixAdapter<M,X,Y> >
    {
    public:
      typedef const MatrixAdapterArgs<M,X,Y> Arguments;

      static inline MatrixAdapter<M,X,Y>* construct(Arguments& args)
      {
        return new MatrixAdapter<M,X,Y>(*args.matrix_);
      }

      static inline void deconstruct(MatrixAdapter<M,X,Y>* m)
      {
        delete m;
      }
    };

    template<>
    class ConstructionTraits<SequentialInformation>
    {
    public:
      typedef const SequentialCommunicationArgs Arguments;
      static inline SequentialInformation* construct(Arguments& args)
      {
        return new SequentialInformation(args.comm_);
      }

      static inline void deconstruct(SequentialInformation* si)
      {
        delete si;
      }
    };


#if HAVE_MPI

    template<class T1, class T2>
    class ConstructionTraits<OwnerOverlapCopyCommunication<T1,T2> >
    {
    public:
      typedef const OwnerOverlapCopyCommunicationArgs Arguments;

      static inline OwnerOverlapCopyCommunication<T1,T2>* construct(Arguments& args)
      {
        return new OwnerOverlapCopyCommunication<T1,T2>(args.comm_, args.cat_);
      }

      static inline void deconstruct(OwnerOverlapCopyCommunication<T1,T2>* com)
      {
        delete com;
      }
    };

#endif

    /** @} */
  } // namespace Amg
} // namespace Dune
#endif
