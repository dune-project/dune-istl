// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_OWNEROVERLAPCOPY_HH
#define DUNE_ISTL_OWNEROVERLAPCOPY_HH

#include <new>
#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <tuple>

#include "cmath"

// MPI header
#if HAVE_MPI
#include <mpi.h>
#endif

#include <dune/common/enumset.hh>

#if HAVE_MPI
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/remoteindices.hh>
#include <dune/common/parallel/mpicommunication.hh>
#endif

#include "solvercategory.hh"
#include "istlexception.hh"
#include <dune/common/parallel/communication.hh>
#include <dune/istl/matrixmarket.hh>

template<int dim, template<class,class> class Comm>
void testRedistributed(int s);


namespace Dune {

  /**
     @addtogroup ISTL_Comm
     @{
   */

  /**
   * @file
   * @brief Classes providing communication interfaces for
   * overlapping Schwarz methods.
   * @author Peter Bastian
   */

  /**
   * @brief Attribute set for overlapping Schwarz.
   */
  struct OwnerOverlapCopyAttributeSet
  {
    enum AttributeSet {
      owner=1, overlap=2, copy=3
    };
  };

  /**
   * @brief Information about the index distribution.
   *
   * This class contains information about indices local to
   * the process together with information about on which
   * processes those indices are also present together with the
   * attribute they have there.
   *
   * This information might be used to set up an IndexSet together with
   * an RemoteIndices object needed for the ISTL communication classes.
   */
  template <class G, class L>
  class IndexInfoFromGrid
  {
  public:
    /** @brief The type of the global index. */
    typedef G GlobalIdType;

    /** @brief The type of the local index. */
    typedef L LocalIdType;

    /**
     * @brief A triple describing a local index.
     *
     * The triple consists of the global index and the local
     * index and an attribute
     */
    typedef std::tuple<GlobalIdType,LocalIdType,int> IndexTripel;
    /**
     * @brief A triple describing a remote index.
     *
     * The triple consists of a process number and the global index and
     * the attribute of the index at the remote process.
     */
    typedef std::tuple<int,GlobalIdType,int> RemoteIndexTripel;

    /**
     * @brief Add a new index triple to the set of local indices.
     *
     * @param x The index triple.
     */
    void addLocalIndex (const IndexTripel& x)
    {
      if (std::get<2>(x)!=OwnerOverlapCopyAttributeSet::owner &&
          std::get<2>(x)!=OwnerOverlapCopyAttributeSet::overlap &&
          std::get<2>(x)!=OwnerOverlapCopyAttributeSet::copy)
        DUNE_THROW(ISTLError,"OwnerOverlapCopyCommunication: global index not in index set");
      localindices.insert(x);
    }

    /**
     * @brief Add a new remote index triple to the set of remote indices.
     *
     * @param x The index triple to add.
     */
    void addRemoteIndex (const RemoteIndexTripel& x)
    {
      if (std::get<2>(x)!=OwnerOverlapCopyAttributeSet::owner &&
          std::get<2>(x)!=OwnerOverlapCopyAttributeSet::overlap &&
          std::get<2>(x)!=OwnerOverlapCopyAttributeSet::copy)
        DUNE_THROW(ISTLError,"OwnerOverlapCopyCommunication: global index not in index set");
      remoteindices.insert(x);
    }

    /**
     * @brief Get the set of indices local to the process.
     * @return The set of local indices.
     */
    const std::set<IndexTripel>& localIndices () const
    {
      return localindices;
    }

    /**
     * @brief Get the set of remote indices.
     * @return the set of remote indices.
     */
    const std::set<RemoteIndexTripel>& remoteIndices () const
    {
      return remoteindices;
    }

    /**
     * @brief Remove all indices from the sets.
     */
    void clear ()
    {
      localindices.clear();
      remoteindices.clear();
    }

  private:
    /** @brief The set of local indices. */
    std::set<IndexTripel> localindices;
    /** @brief The set of remote indices. */
    std::set<RemoteIndexTripel> remoteindices;
  };


#if HAVE_MPI

  /**
   * @brief A class setting up standard communication for a two-valued
   * attribute set with owner/overlap/copy semantics.
   *
   * set up communication from known distribution with owner/overlap/copy semantics
   */
  template <class GlobalIdType, class LocalIdType=int>
  class OwnerOverlapCopyCommunication
  {
    template<typename M, typename G, typename L>
    friend void loadMatrixMarket(M&,
                                 const std::string&,
                                 OwnerOverlapCopyCommunication<G,L>&,
                                 bool);
    // used types
    typedef typename IndexInfoFromGrid<GlobalIdType,LocalIdType>::IndexTripel IndexTripel;
    typedef typename IndexInfoFromGrid<GlobalIdType,LocalIdType>::RemoteIndexTripel RemoteIndexTripel;
    typedef typename std::set<IndexTripel>::const_iterator localindex_iterator;
    typedef typename std::set<RemoteIndexTripel>::const_iterator remoteindex_iterator;
    typedef typename OwnerOverlapCopyAttributeSet::AttributeSet AttributeSet;
    typedef Dune::ParallelLocalIndex<AttributeSet> LI;
  public:
    typedef Dune::ParallelIndexSet<GlobalIdType,LI,512> PIS;
    typedef Dune::RemoteIndices<PIS> RI;
    typedef Dune::RemoteIndexListModifier<PIS,typename RI::Allocator,false> RILM;
    typedef typename RI::RemoteIndex RX;
    typedef Dune::BufferedCommunicator BC;
    typedef Dune::Interface IF;
    typedef EnumItem<AttributeSet,OwnerOverlapCopyAttributeSet::owner> OwnerSet;
    typedef EnumItem<AttributeSet,OwnerOverlapCopyAttributeSet::copy> CopySet;
    typedef Combine<EnumItem<AttributeSet,OwnerOverlapCopyAttributeSet::owner>,EnumItem<AttributeSet,OwnerOverlapCopyAttributeSet::overlap>,AttributeSet> OwnerOverlapSet;
    typedef Dune::AllSet<AttributeSet> AllSet;
  protected:


    /** \brief gather/scatter callback for communcation */
    template<typename T>
    struct CopyGatherScatter
    {
      typedef typename CommPolicy<T>::IndexedType V;

      static V gather(const T& a, std::size_t i)
      {
        return a[i];
      }

      static void scatter(T& a, V v, std::size_t i)
      {
        a[i] = v;
      }
    };
    template<typename T>
    struct AddGatherScatter
    {
      typedef typename CommPolicy<T>::IndexedType V;

      static V gather(const T& a, std::size_t i)
      {
        return a[i];
      }

      static void scatter(T& a, V v, std::size_t i)
      {
        a[i] += v;
      }
    };

    void buildOwnerOverlapToAllInterface () const
    {
      if (OwnerOverlapToAllInterfaceBuilt)
        OwnerOverlapToAllInterface.free();
      OwnerOverlapSet sourceFlags;
      Combine<OwnerOverlapSet,EnumItem<AttributeSet,OwnerOverlapCopyAttributeSet::copy>,AttributeSet>
        destFlags;
      OwnerOverlapToAllInterface.build(ri,sourceFlags,destFlags);
      OwnerOverlapToAllInterfaceBuilt = true;
    }

    void buildOwnerToAllInterface () const
    {
      if (OwnerToAllInterfaceBuilt)
        OwnerToAllInterface.free();
      OwnerSet sourceFlags;
      AllSet destFlags;
      OwnerToAllInterface.build(ri,sourceFlags,destFlags);
      OwnerToAllInterfaceBuilt = true;
    }

    void buildOwnerCopyToAllInterface () const
    {
      if (OwnerCopyToAllInterfaceBuilt)
        OwnerCopyToAllInterface.free();

      typedef Combine<EnumItem<AttributeSet,OwnerOverlapCopyAttributeSet::owner>,EnumItem<AttributeSet,OwnerOverlapCopyAttributeSet::copy>,AttributeSet> OwnerCopySet;
      OwnerCopySet sourceFlags;
      Combine<OwnerCopySet,EnumItem<AttributeSet,OwnerOverlapCopyAttributeSet::overlap>,AttributeSet> destFlags;
      OwnerCopyToAllInterface.build(ri,sourceFlags,destFlags);
      OwnerCopyToAllInterfaceBuilt = true;
    }

    void buildOwnerCopyToOwnerCopyInterface () const
    {
      if (OwnerCopyToOwnerCopyInterfaceBuilt)
        OwnerCopyToOwnerCopyInterface.free();


      typedef Combine<EnumItem<AttributeSet,OwnerOverlapCopyAttributeSet::owner>,EnumItem<AttributeSet,OwnerOverlapCopyAttributeSet::copy>,AttributeSet> OwnerCopySet;
      OwnerCopySet sourceFlags;
      OwnerCopySet destFlags;
      OwnerCopyToOwnerCopyInterface.build(ri,sourceFlags,destFlags);
      OwnerCopyToOwnerCopyInterfaceBuilt = true;
    }

    void buildCopyToAllInterface () const
    {
      if (CopyToAllInterfaceBuilt)
        CopyToAllInterface.free();
      CopySet sourceFlags;
      AllSet destFlags;
      CopyToAllInterface.build(ri,sourceFlags,destFlags);
      CopyToAllInterfaceBuilt = true;
    }

  public:

    /**
     * @brief Get Solver Category.
     * @return The Solver Category.
     */
    SolverCategory::Category category () const {
      return category_;
    }

    const Communication<MPI_Comm>& communicator() const
    {
      return cc;
    }

    /**
     * @brief Communicate values from owner data points to all other data points.
     *
     * @brief source The data to send from.
     * @brief dest The data to send to.
     */
    template<class T>
    void copyOwnerToAll (const T& source, T& dest) const
    {
      if (!OwnerToAllInterfaceBuilt)
        buildOwnerToAllInterface ();
      BC communicator;
      communicator.template build<T>(OwnerToAllInterface);
      communicator.template forward<CopyGatherScatter<T> >(source,dest);
      communicator.free();
    }

    /**
     * @brief Communicate values from copy data points to all other data points.
     *
     * @brief source The data to send from.
     * @brief dest The data to send to.
     */
    template<class T>
    void copyCopyToAll (const T& source, T& dest) const
    {
      if (!CopyToAllInterfaceBuilt)
        buildCopyToAllInterface ();
      BC communicator;
      communicator.template build<T>(CopyToAllInterface);
      communicator.template forward<CopyGatherScatter<T> >(source,dest);
      communicator.free();
    }

    /**
     * @brief Communicate values from owner data points to all other data points and add them to those values.
     *
     * @brief source The data to send from.
     * @brief dest The data to add them communicated values to.
     */
    template<class T>
    void addOwnerOverlapToAll (const T& source, T& dest) const
    {
      if (!OwnerOverlapToAllInterfaceBuilt)
        buildOwnerOverlapToAllInterface ();
      BC communicator;
      communicator.template build<T>(OwnerOverlapToAllInterface);
      communicator.template forward<AddGatherScatter<T> >(source,dest);
      communicator.free();
    }

    /**
     * @brief Communicate values from owner and copy data points to all other data points and add them to those values.
     *
     * @brief source The data to send from.
     * @brief dest The data to add them communicated values to.
     */
    template<class T>
    void addOwnerCopyToAll (const T& source, T& dest) const
    {
      if (!OwnerCopyToAllInterfaceBuilt)
        buildOwnerCopyToAllInterface ();
      BC communicator;
      communicator.template build<T>(OwnerCopyToAllInterface);
      communicator.template forward<AddGatherScatter<T> >(source,dest);
      communicator.free();
    }

    /**
     * @brief Communicate values from owner and copy data points to owner and copy data points and add them to those values.
     *
     * @brief source The data to send from.
     * @brief dest The data to add the communicated values to.
     */
    template<class T>
    void addOwnerCopyToOwnerCopy (const T& source, T& dest) const
    {
      if (!OwnerCopyToOwnerCopyInterfaceBuilt)
        buildOwnerCopyToOwnerCopyInterface ();
      BC communicator;
      communicator.template build<T>(OwnerCopyToOwnerCopyInterface);
      communicator.template forward<AddGatherScatter<T> >(source,dest);
      communicator.free();
    }


    /**
     * @brief Compute a global dot product of two vectors.
     *
     * @param x The first vector of the product.
     * @param y The second vector of the product.
     * @param result Reference to store the result in.
     */
    template<class T1, class T2>
    void dot (const T1& x, const T1& y, T2& result) const
    {
      using real_type = typename FieldTraits<typename T1::field_type>::real_type;
      // set up mask vector
      if (mask.size()!=static_cast<typename std::vector<double>::size_type>(x.size()))
      {
        mask.resize(x.size());
        for (typename std::vector<double>::size_type i=0; i<mask.size(); i++)
          mask[i] = 1;
        for (typename PIS::const_iterator i=pis.begin(); i!=pis.end(); ++i)
          if (i->local().attribute()!=OwnerOverlapCopyAttributeSet::owner)
            mask[i->local().local()] = 0;
      }
      result = T2(0.0);

      for (typename T1::size_type i=0; i<x.size(); i++)
        result += (x[i]*(y[i]))*static_cast<real_type>(mask[i]);
      result = cc.sum(result);
    }

    /**
     * @brief Compute the global Euclidean norm of a vector.
     *
     * @param x The vector to compute the norm of.
     * @return The global Euclidean norm of that vector.
     */
    template<class T1>
    typename FieldTraits<typename T1::field_type>::real_type norm (const T1& x) const
    {
      using real_type = typename FieldTraits<typename T1::field_type>::real_type;

      // set up mask vector
      if (mask.size()!=static_cast<typename std::vector<double>::size_type>(x.size()))
      {
        mask.resize(x.size());
        for (typename std::vector<double>::size_type i=0; i<mask.size(); i++)
          mask[i] = 1;
        for (typename PIS::const_iterator i=pis.begin(); i!=pis.end(); ++i)
          if (i->local().attribute()!=OwnerOverlapCopyAttributeSet::owner)
            mask[i->local().local()] = 0;
      }
      auto result = real_type(0.0);
      for (typename T1::size_type i=0; i<x.size(); i++)
        result += Impl::asVector(x[i]).two_norm2()*mask[i];
      using std::sqrt;
      return sqrt(cc.sum(result));
    }

    typedef Dune::EnumItem<AttributeSet,OwnerOverlapCopyAttributeSet::copy> CopyFlags;

    /** @brief The type of the parallel index set. */
    typedef Dune::ParallelIndexSet<GlobalIdType,LI,512> ParallelIndexSet;

    /** @brief The type of the remote indices. */
    typedef Dune::RemoteIndices<PIS> RemoteIndices;

    /**
     * @brief The type of the reverse lookup of indices. */
    typedef Dune::GlobalLookupIndexSet<ParallelIndexSet> GlobalLookupIndexSet;

    /**
     * @brief Get the underlying parallel index set.
     * @return The underlying parallel index set.
     */
    const ParallelIndexSet& indexSet() const
    {
      return pis;
    }

    /**
     * @brief Get the underlying remote indices.
     * @return The underlying remote indices.
     */
    const RemoteIndices& remoteIndices() const
    {
      return ri;
    }

    /**
     * @brief Get the underlying parallel index set.
     * @return The underlying parallel index set.
     */
    ParallelIndexSet& indexSet()
    {
      return pis;
    }


    /**
     * @brief Get the underlying remote indices.
     * @return The underlying remote indices.
     */
    RemoteIndices& remoteIndices()
    {
      return ri;
    }

    void buildGlobalLookup()
    {
      if(globalLookup_) {
        if(pis.seqNo()==oldseqNo)
          // Nothing changed!
          return;
        delete globalLookup_;
      }

      globalLookup_ = new GlobalLookupIndexSet(pis);
      oldseqNo = pis.seqNo();
    }

    void buildGlobalLookup(std::size_t size)
    {
      if(globalLookup_) {
        if(pis.seqNo()==oldseqNo)
          // Nothing changed!
          return;
        delete globalLookup_;
      }
      globalLookup_ = new GlobalLookupIndexSet(pis, size);
      oldseqNo = pis.seqNo();
    }

    void freeGlobalLookup()
    {
      delete globalLookup_;
      globalLookup_=0;
    }

    const GlobalLookupIndexSet& globalLookup() const
    {
      assert(globalLookup_ != 0);
      return *globalLookup_;
    }

    /**
     * @brief Set vector to zero at copy dofs
     *
     * @param x The vector to project.
     */
    template<class T1>
    void project (T1& x) const
    {
      for (typename PIS::const_iterator i=pis.begin(); i!=pis.end(); ++i)
        if (i->local().attribute()==OwnerOverlapCopyAttributeSet::copy)
          x[i->local().local()] = 0;
    }

    /**
     * @brief Construct the communication without any indices.
     *
     * The local index set and the remote indices have to be set up
     * later on.
     * @param comm_ The MPI Communicator to use, e. g. MPI_COMM_WORLD
     * @param cat_ The Solver category, default is overlapping
     * @param freecomm_ Whether to free the communicator comm_ in the destructor, default is false
     */
    OwnerOverlapCopyCommunication (MPI_Comm comm_,
                                   SolverCategory::Category cat_ = SolverCategory::overlapping,
                                   bool freecomm_ = false)
      : comm(comm_), cc(comm_), pis(), ri(pis,pis,comm_),
        OwnerToAllInterfaceBuilt(false), OwnerOverlapToAllInterfaceBuilt(false),
        OwnerCopyToAllInterfaceBuilt(false), OwnerCopyToOwnerCopyInterfaceBuilt(false),
        CopyToAllInterfaceBuilt(false), globalLookup_(0), category_(cat_),
        freecomm(freecomm_)
    {}

    /**
     * @brief Construct the communication without any indices using MPI_COMM_WORLD.
     *
     * The local index set and the remote indices have to be set up
     * later on.
     * @param cat_ The Solver category, default is overlapping
       is false
     */
    OwnerOverlapCopyCommunication (SolverCategory::Category cat_ = SolverCategory::overlapping)
      : comm(MPI_COMM_WORLD), cc(MPI_COMM_WORLD), pis(), ri(pis,pis,MPI_COMM_WORLD),
        OwnerToAllInterfaceBuilt(false), OwnerOverlapToAllInterfaceBuilt(false),
        OwnerCopyToAllInterfaceBuilt(false), OwnerCopyToOwnerCopyInterfaceBuilt(false),
        CopyToAllInterfaceBuilt(false), globalLookup_(0), category_(cat_), freecomm(false)
    {}

    /**
     * @brief Constructor
     * @param indexinfo The set of IndexTripels describing the local and remote indices.
     * @param comm_ The communicator to use in the communication.
     * @param cat_ The Solver category, default is overlapping
     * @param freecomm_ Whether to free the communicator comm_ in the destructor, default is false
     */
    OwnerOverlapCopyCommunication (const IndexInfoFromGrid<GlobalIdType, LocalIdType>& indexinfo,
                                   MPI_Comm comm_,
                                   SolverCategory::Category cat_ = SolverCategory::overlapping,
                                   bool freecomm_ = false)
      : comm(comm_), cc(comm_), OwnerToAllInterfaceBuilt(false),
        OwnerOverlapToAllInterfaceBuilt(false), OwnerCopyToAllInterfaceBuilt(false),
        OwnerCopyToOwnerCopyInterfaceBuilt(false), CopyToAllInterfaceBuilt(false),
        globalLookup_(0), category_(cat_), freecomm(freecomm_)
    {
      // set up an ISTL index set
      pis.beginResize();
      for (localindex_iterator i=indexinfo.localIndices().begin(); i!=indexinfo.localIndices().end(); ++i)
      {
        if (std::get<2>(*i)==OwnerOverlapCopyAttributeSet::owner)
          pis.add(std::get<0>(*i),LI(std::get<1>(*i),OwnerOverlapCopyAttributeSet::owner,true));
        if (std::get<2>(*i)==OwnerOverlapCopyAttributeSet::overlap)
          pis.add(std::get<0>(*i),LI(std::get<1>(*i),OwnerOverlapCopyAttributeSet::overlap,true));
        if (std::get<2>(*i)==OwnerOverlapCopyAttributeSet::copy)
          pis.add(std::get<0>(*i),LI(std::get<1>(*i),OwnerOverlapCopyAttributeSet::copy,true));
        //                std::cout << cc.rank() << ": adding index " << std::get<0>(*i) << " " << std::get<1>(*i) << " " << std::get<2>(*i) << std::endl;
      }
      pis.endResize();

      // build remote indices WITHOUT communication
      //          std::cout << cc.rank() << ": build remote indices" << std::endl;
      ri.setIndexSets(pis,pis,cc);
      if (indexinfo.remoteIndices().size()>0)
      {
        remoteindex_iterator i=indexinfo.remoteIndices().begin();
        int p = std::get<0>(*i);
        RILM modifier = ri.template getModifier<false,true>(p);
        typename PIS::const_iterator pi=pis.begin();
        for ( ; i!=indexinfo.remoteIndices().end(); ++i)
        {
          // handle processor change
          if (p!=std::get<0>(*i))
          {
            p = std::get<0>(*i);
            modifier = ri.template getModifier<false,true>(p);
            pi=pis.begin();
          }

          // position to correct entry in parallel index set
          while (pi->global()!=std::get<1>(*i) && pi!=pis.end())
            ++pi;
          if (pi==pis.end())
            DUNE_THROW(ISTLError,"OwnerOverlapCopyCommunication: global index not in index set");

          // insert entry
          //                      std::cout << cc.rank() << ": adding remote index " << std::get<0>(*i) << " " << std::get<1>(*i) << " " << std::get<2>(*i) << std::endl;
          if (std::get<2>(*i)==OwnerOverlapCopyAttributeSet::owner)
            modifier.insert(RX(OwnerOverlapCopyAttributeSet::owner,&(*pi)));
          if (std::get<2>(*i)==OwnerOverlapCopyAttributeSet::overlap)
            modifier.insert(RX(OwnerOverlapCopyAttributeSet::overlap,&(*pi)));
          if (std::get<2>(*i)==OwnerOverlapCopyAttributeSet::copy)
            modifier.insert(RX(OwnerOverlapCopyAttributeSet::copy,&(*pi)));
        }
      }else{
        // Force remote indices to be synced!
        ri.template getModifier<false,true>(0);
      }
    }

    // destructor: free memory in some objects
    ~OwnerOverlapCopyCommunication ()
    {
      ri.free();
      if (OwnerToAllInterfaceBuilt) OwnerToAllInterface.free();
      if (OwnerOverlapToAllInterfaceBuilt) OwnerOverlapToAllInterface.free();
      if (OwnerCopyToAllInterfaceBuilt) OwnerCopyToAllInterface.free();
      if (OwnerCopyToOwnerCopyInterfaceBuilt) OwnerCopyToOwnerCopyInterface.free();
      if (CopyToAllInterfaceBuilt) CopyToAllInterface.free();
      if (globalLookup_) delete globalLookup_;
      if (freecomm==true)
        if(comm!=MPI_COMM_NULL)
        {
#ifdef MPI_2
          // If it is possible to query whether MPI_Finalize
          // was called, only free the communicator before
          // calling MPI_Finalize.
          int wasFinalized = 0;
          MPI_Finalized( &wasFinalized );
          if(!wasFinalized)
#endif
          MPI_Comm_free(&comm);
        }
    }

  private:
    OwnerOverlapCopyCommunication (const OwnerOverlapCopyCommunication&)
    {}
    MPI_Comm comm;
    Communication<MPI_Comm> cc;
    PIS pis;
    RI ri;
    mutable IF OwnerToAllInterface;
    mutable bool OwnerToAllInterfaceBuilt;
    mutable IF OwnerOverlapToAllInterface;
    mutable bool OwnerOverlapToAllInterfaceBuilt;
    mutable IF OwnerCopyToAllInterface;
    mutable bool OwnerCopyToAllInterfaceBuilt;
    mutable IF OwnerCopyToOwnerCopyInterface;
    mutable bool OwnerCopyToOwnerCopyInterfaceBuilt;
    mutable IF CopyToAllInterface;
    mutable bool CopyToAllInterfaceBuilt;
    mutable std::vector<double> mask;
    int oldseqNo;
    GlobalLookupIndexSet* globalLookup_;
    const SolverCategory::Category category_;
    bool freecomm;
  };

#endif


  /** @} end documentation */

} // end namespace

#endif
