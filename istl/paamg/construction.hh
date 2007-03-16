// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_AMGCONSTRUCTION_HH
#define DUNE_AMGCONSTRUCTION_HH

#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/owneroverlapcopy.hh>
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
       * @param object Pointer to the space allocated for the object.
       * @param arguments The arguments for the construction.
       */
      static inline T* construct(Arguments&  args)
      {
        return new T();
      }

      static inline void deconstruct(T* t)
      {
        delete t;
      }

    };

    template<class T>
    class ConstructionTraits<BlockVector<T> >
    {
    public:
      typedef const int Arguments;
      static inline BlockVector<T>* construct(Arguments& n)
      {
        return new BlockVector<T>(n);
      }

      static inline void deconstruct(BlockVector<T>* t)
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

  } // end Amg namspace

  // foward declaration
  template<class M, class X, class Y, class C>
  class OverlappingSchwarzOperator;

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
      typedef const CollectiveCommunication<void*> Arguments;

      static inline SequentialInformation* construct(Arguments& args)
      {
        return new SequentialInformation(args);
      }

      static inline void deconstruct(SequentialInformation* si)
      {
        delete si;
      }
    };


#if HAVE_MPI
    template<typename T>
    class ConstructionTraits<ParallelInformation<T> >
    {
    public:
      typedef const MPI_Comm Arguments;

      static inline ParallelInformation<T>* construct(Arguments& args)
      {
        return new ParallelInformation<T>(args);
      }

      static inline void deconstruct(ParallelInformation<T>* pi)
      {
        delete pi;
      }
    };


    template<class T1, class T2>
    class ConstructionTraits<OwnerOverlapCopyCommunication<T1,T2> >
    {
    public:
      typedef const MPI_Comm Arguments;

      static inline OwnerOverlapCopyCommunication<T1,T2>* construct(Arguments& args)
      {
        return new OwnerOverlapCopyCommunication<T1,T2>(args);
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
