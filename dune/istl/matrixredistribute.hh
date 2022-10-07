// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_MATRIXREDISTRIBUTE_HH
#define DUNE_ISTL_MATRIXREDISTRIBUTE_HH
#include <memory>
#include "repartition.hh"
#include <dune/common/exceptions.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/paamg/pinfo.hh>
/**
 * @file
 * @brief Functionality for redistributing a sparse matrix.
 * @author Markus Blatt
 */
namespace Dune
{
  template<typename  T>
  struct RedistributeInformation
  {
    bool isSetup() const
    {
      return false;
    }
    template<class D>
    void redistribute([[maybe_unused]] const D& from, [[maybe_unused]] D& to) const
    {}

    template<class D>
    void redistributeBackward([[maybe_unused]] D& from, [[maybe_unused]]const D& to) const
    {}

    void resetSetup()
    {}

    void setNoRows([[maybe_unused]] std::size_t size)
    {}

    void setNoCopyRows([[maybe_unused]] std::size_t size)
    {}

    void setNoBackwardsCopyRows([[maybe_unused]] std::size_t size)
    {}

    std::size_t getRowSize([[maybe_unused]] std::size_t index) const
    {
      return -1;
    }

    std::size_t getCopyRowSize([[maybe_unused]] std::size_t index) const
    {
      return -1;
    }

    std::size_t getBackwardsCopyRowSize([[maybe_unused]] std::size_t index) const
    {
      return -1;
    }

  };

#if HAVE_MPI
  template<typename  T, typename T1>
  class RedistributeInformation<OwnerOverlapCopyCommunication<T,T1> >
  {
  public:
    typedef OwnerOverlapCopyCommunication<T,T1> Comm;

    RedistributeInformation()
      : interface(), setup_(false)
    {}

    RedistributeInterface& getInterface()
    {
      return interface;
    }
    template<typename IS>
    void checkInterface(const IS& source,
                        const IS& target, MPI_Comm comm)
    {
      auto ri = std::make_unique<RemoteIndices<IS> >(source, target, comm);
      ri->template rebuild<true>();
      Interface inf;
      typename OwnerOverlapCopyCommunication<int>::OwnerSet flags;
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      inf.free();
      inf.build(*ri, flags, flags);


#ifdef DEBUG_REPART
      if(inf!=interface) {

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
          std::cout<<"Interfaces do not match!"<<std::endl;
        std::cout<<rank<<": redist interface new :"<<inf<<std::endl;
        std::cout<<rank<<": redist interface :"<<interface<<std::endl;

        throw "autsch!";
      }
#endif
    }
    void setSetup()
    {
      setup_=true;
      interface.strip();
    }

    void resetSetup()
    {
      setup_=false;
    }

    template<class GatherScatter, class D>
    void redistribute(const D& from, D& to) const
    {
      BufferedCommunicator communicator;
      communicator.template build<D>(from,to, interface);
      communicator.template forward<GatherScatter>(from, to);
      communicator.free();
    }
    template<class GatherScatter, class D>
    void redistributeBackward(D& from, const D& to) const
    {

      BufferedCommunicator communicator;
      communicator.template build<D>(from,to, interface);
      communicator.template backward<GatherScatter>(from, to);
      communicator.free();
    }

    template<class D>
    void redistribute(const D& from, D& to) const
    {
      redistribute<CopyGatherScatter<D> >(from,to);
    }
    template<class D>
    void redistributeBackward(D& from, const D& to) const
    {
      redistributeBackward<CopyGatherScatter<D> >(from,to);
    }
    bool isSetup() const
    {
      return setup_;
    }

    void reserve(std::size_t size)
    {}

    std::size_t& getRowSize(std::size_t index)
    {
      return rowSize[index];
    }

    std::size_t getRowSize(std::size_t index) const
    {
      return rowSize[index];
    }

    std::size_t& getCopyRowSize(std::size_t index)
    {
      return copyrowSize[index];
    }

    std::size_t getCopyRowSize(std::size_t index) const
    {
      return copyrowSize[index];
    }

    std::size_t& getBackwardsCopyRowSize(std::size_t index)
    {
      return backwardscopyrowSize[index];
    }

    std::size_t getBackwardsCopyRowSize(std::size_t index) const
    {
      return backwardscopyrowSize[index];
    }

    void setNoRows(std::size_t rows)
    {
      rowSize.resize(rows, 0);
    }

    void setNoCopyRows(std::size_t rows)
    {
      copyrowSize.resize(rows, 0);
    }

    void setNoBackwardsCopyRows(std::size_t rows)
    {
      backwardscopyrowSize.resize(rows, 0);
    }

  private:
    std::vector<std::size_t> rowSize;
    std::vector<std::size_t> copyrowSize;
    std::vector<std::size_t> backwardscopyrowSize;
    RedistributeInterface interface;
    bool setup_;
  };

  /**
   * @brief Utility class to communicate and set the row sizes
   * of a redistributed matrix.
   *
   * @tparam M The type of the matrix that the row size
   * is communicated of.
   * @tparam RI The type of class for redistribution information
   */
  template<class M, class RI>
  struct CommMatrixRowSize
  {
    // Make the default communication policy work.
    typedef typename M::size_type value_type;
    typedef typename M::size_type size_type;

    /**
     * @brief Constructor.
     * @param m_ The matrix whose sparsity pattern is communicated.
     * @param[out] rowsize_ RedistributeInformation object
     */
    CommMatrixRowSize(const M& m_, RI& rowsize_)
      : matrix(m_), rowsize(rowsize_)
    {}
    const M& matrix;
    RI& rowsize;

  };


  /**
   * @brief Utility class to communicate and build the sparsity pattern
   * of a redistributed matrix.
   *
   * @tparam M The type of the matrix that the sparsity pattern
   * is communicated of.
   * @tparam I The type of the index set.
   */
  template<class M, class I>
  struct CommMatrixSparsityPattern
  {
    typedef typename M::size_type size_type;

    /**
     * @brief Constructor for the original side
     * @param m_ The matrix whose sparsity pattern is communicated.
     * @param idxset_ The index set corresponding to the local matrix.
     * @param aggidxset_ The index set corresponding to the redistributed matrix.
     */
    CommMatrixSparsityPattern(const M& m_, const Dune::GlobalLookupIndexSet<I>& idxset_, const I& aggidxset_)
      : matrix(m_), idxset(idxset_), aggidxset(aggidxset_), rowsize()
    {}

    /**
     * @brief Constructor for the redistruted side.
     * @param m_ The matrix whose sparsity pattern is communicated.
     * @param idxset_ The index set corresponding to the local matrix.
     * @param aggidxset_ The index set corresponding to the redistributed matrix.
     * @param rowsize_ The row size for the redistributed owner rows.
     */
    CommMatrixSparsityPattern(const M& m_, const Dune::GlobalLookupIndexSet<I>& idxset_, const I& aggidxset_,
                              const std::vector<typename M::size_type>& rowsize_)
      : matrix(m_), idxset(idxset_), aggidxset(aggidxset_), sparsity(aggidxset_.size()), rowsize(&rowsize_)
    {}

    /**
     * @brief Creates and stores the sparsity pattern of the redistributed matrix.
     *
     * After the pattern is communicated this function can be used.
     * @param m The matrix to build.
     */
    void storeSparsityPattern(M& m)
    {
      // insert diagonal to overlap rows
      typedef typename Dune::GlobalLookupIndexSet<I>::const_iterator IIter;
      typedef typename Dune::OwnerOverlapCopyCommunication<int>::OwnerSet OwnerSet;
      std::size_t nnz=0;
#ifdef DEBUG_REPART
      int rank;

      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
      for(IIter i= aggidxset.begin(), end=aggidxset.end(); i!=end; ++i) {
        if(!OwnerSet::contains(i->local().attribute())) {
#ifdef DEBUG_REPART
          std::cout<<rank<<" Inserting diagonal for"<<i->local()<<std::endl;
#endif
          sparsity[i->local()].insert(i->local());
        }

        nnz+=sparsity[i->local()].size();
      }
      assert( aggidxset.size()==sparsity.size());

      if(nnz>0) {
        m.setSize(aggidxset.size(), aggidxset.size(), nnz);
        m.setBuildMode(M::row_wise);
        typename M::CreateIterator citer=m.createbegin();
#ifdef DEBUG_REPART
        std::size_t idx=0;
        bool correct=true;
        Dune::GlobalLookupIndexSet<I> global(aggidxset);
#endif
        typedef typename std::vector<std::set<size_type> >::const_iterator Iter;
        for(Iter i=sparsity.begin(), end=sparsity.end(); i!=end; ++i, ++citer)
        {
          typedef typename std::set<size_type>::const_iterator SIter;
          for(SIter si=i->begin(), send=i->end(); si!=send; ++si)
            citer.insert(*si);
#ifdef DEBUG_REPART
          if(i->find(idx)==i->end()) {
            const typename I::IndexPair* gi=global.pair(idx);
            assert(gi);
            std::cout<<rank<<": row "<<idx<<" is missing a diagonal entry! global="<<gi->global()<<" attr="<<gi->local().attribute()<<" "<<
            OwnerSet::contains(gi->local().attribute())<<
            " row size="<<i->size()<<std::endl;
            correct=false;
          }
          ++idx;
#endif
        }
#ifdef DEBUG_REPART
        if(!correct)
          throw "bla";
#endif
      }
    }

    /**
     * @brief Completes the sparsity pattern of the redistributed matrix with data
     * from copy rows for the novlp case.
     *
     * After the pattern communication this function can be used.
     * @param add_sparsity Sparsity pattern from the copy rows.
     */
    void completeSparsityPattern(std::vector<std::set<size_type> > add_sparsity)
    {
      for (unsigned int i = 0; i != sparsity.size(); ++i) {
        if (add_sparsity[i].size() != 0) {
          typedef std::set<size_type> Set;
          Set tmp_set;
          std::insert_iterator<Set> tmp_insert (tmp_set, tmp_set.begin());
          std::set_union(add_sparsity[i].begin(), add_sparsity[i].end(),
                         sparsity[i].begin(), sparsity[i].end(), tmp_insert);
          sparsity[i].swap(tmp_set);
        }
      }
    }

    const M& matrix;
    typedef Dune::GlobalLookupIndexSet<I> LookupIndexSet;
    const Dune::GlobalLookupIndexSet<I>& idxset;
    const I& aggidxset;
    std::vector<std::set<size_type> > sparsity;
    const std::vector<size_type>* rowsize;
  };

  template<class M, class I>
  struct CommPolicy<CommMatrixSparsityPattern<M,I> >
  {
    typedef CommMatrixSparsityPattern<M,I> Type;

    /**
     *  @brief The indexed type we send.
     * This is the global index indentitfying the column.
     */
    typedef typename I::GlobalIndex IndexedType;

    /** @brief Each row varies in size. */
    typedef VariableSize IndexedTypeFlag;

    static typename M::size_type getSize(const Type& t, std::size_t i)
    {
      if(!t.rowsize)
        return t.matrix[i].size();
      else
      {
        assert((*t.rowsize)[i]>0);
        return (*t.rowsize)[i];
      }
    }
  };

  /**
   * @brief Utility class for comunicating the matrix entries.
   *
   * @tparam M The type of the matrix.
   * @tparam I The type of the ParallelIndexSet.
   */
  template<class M, class I>
  struct CommMatrixRow
  {
    /**
     * @brief Constructor.
     * @param m_ The matrix to communicate the values. That is the local original matrix
     * as the source of the communication and the redistributed at the target of the
     * communication.
     * @param idxset_ The index set for the original matrix.
     * @param aggidxset_ The index set for the redistributed matrix.
     */
    CommMatrixRow(M& m_, const Dune::GlobalLookupIndexSet<I>& idxset_, const I& aggidxset_)
      : matrix(m_), idxset(idxset_), aggidxset(aggidxset_), rowsize()
    {}

    /**
     * @brief Constructor.
     */
    CommMatrixRow(M& m_, const Dune::GlobalLookupIndexSet<I>& idxset_, const I& aggidxset_,
                  std::vector<size_t>& rowsize_)
      : matrix(m_), idxset(idxset_), aggidxset(aggidxset_), rowsize(&rowsize_)
    {}
    /**
     * @brief Sets the non-owner rows correctly as Dirichlet boundaries.
     *
     * This should be called after the communication.
     */
    void setOverlapRowsToDirichlet()
    {
      typedef typename Dune::GlobalLookupIndexSet<I>::const_iterator Iter;
      typedef typename Dune::OwnerOverlapCopyCommunication<int>::OwnerSet OwnerSet;

      for(Iter i= aggidxset.begin(), end=aggidxset.end(); i!=end; ++i)
        if(!OwnerSet::contains(i->local().attribute())) {
          // Set to Dirchlet
          typedef typename M::ColIterator CIter;
          for(CIter c=matrix[i->local()].begin(), cend= matrix[i->local()].end();
              c!= cend; ++c)
          {
            *c=0;
            if(c.index()==i->local()) {
              auto setDiagonal = [](auto&& scalarOrMatrix, const auto& value) {
                auto&& matrixView = Dune::Impl::asMatrix(scalarOrMatrix);
                for (auto rowIt = matrixView.begin(); rowIt != matrixView.end(); ++rowIt)
                  (*rowIt)[rowIt.index()] = value;
              };
              setDiagonal(*c, 1);
            }
          }
        }
    }
    /** @brief The matrix to communicate the values of. */
    M& matrix;
    /** @brief Index set for the original matrix. */
    const Dune::GlobalLookupIndexSet<I>& idxset;
    /** @brief Index set for the redistributed matrix. */
    const I& aggidxset;
    /** @brief row size information for the receiving side. */
    std::vector<size_t>* rowsize; // row sizes differ from sender side in overlap!
  };

  template<class M, class I>
  struct CommPolicy<CommMatrixRow<M,I> >
  {
    typedef CommMatrixRow<M,I> Type;

    /**
     *  @brief The indexed type we send.
     * This is the pair of global index indentitfying the column and the value itself.
     */
    typedef std::pair<typename I::GlobalIndex,typename M::block_type> IndexedType;

    /** @brief Each row varies in size. */
    typedef VariableSize IndexedTypeFlag;

    static std::size_t getSize(const Type& t, std::size_t i)
    {
      if(!t.rowsize)
        return t.matrix[i].size();
      else
      {
        assert((*t.rowsize)[i]>0);
        return (*t.rowsize)[i];
      }
    }
  };

  template<class M, class I, class RI>
  struct MatrixRowSizeGatherScatter
  {
    typedef CommMatrixRowSize<M,RI> Container;

    static const typename M::size_type gather(const Container& cont, std::size_t i)
    {
      return cont.matrix[i].size();
    }
    static void scatter(Container& cont, const typename M::size_type& rowsize, std::size_t i)
    {
      assert(rowsize);
      cont.rowsize.getRowSize(i)=rowsize;
    }

  };

  template<class M, class I, class RI>
  struct MatrixCopyRowSizeGatherScatter
  {
    typedef CommMatrixRowSize<M,RI> Container;

    static const typename M::size_type gather(const Container& cont, std::size_t i)
    {
      return cont.matrix[i].size();
    }
    static void scatter(Container& cont, const typename M::size_type& rowsize, std::size_t i)
    {
      assert(rowsize);
      if (rowsize > cont.rowsize.getCopyRowSize(i))
        cont.rowsize.getCopyRowSize(i)=rowsize;
    }

  };

  template<class M, class I>
  struct MatrixSparsityPatternGatherScatter
  {
    typedef typename I::GlobalIndex GlobalIndex;
    typedef CommMatrixSparsityPattern<M,I> Container;
    typedef typename M::ConstColIterator ColIter;

    static ColIter col;
    static GlobalIndex numlimits;

    static const GlobalIndex& gather(const Container& cont, std::size_t i, std::size_t j)
    {
      if(j==0)
        col=cont.matrix[i].begin();
      else if (col!=cont.matrix[i].end())
        ++col;

      //copy communication: different row sizes for copy rows with the same global index
      //are possible. If all values of current  matrix row are sent, send
      //std::numeric_limits<GlobalIndex>::max()
      //and receiver will ignore it.
      if (col==cont.matrix[i].end()) {
        numlimits = std::numeric_limits<GlobalIndex>::max();
        return numlimits;
      }
      else {
        const typename I::IndexPair* index=cont.idxset.pair(col.index());
        assert(index);
        // Only send index if col is no ghost
        if ( index->local().attribute() != 2)
          return index->global();
        else {
          numlimits = std::numeric_limits<GlobalIndex>::max();
          return numlimits;
        }
      }
    }
    static void scatter(Container& cont, const GlobalIndex& gi, std::size_t i, [[maybe_unused]] std::size_t j)
    {
      try{
        if (gi != std::numeric_limits<GlobalIndex>::max()) {
          const typename I::IndexPair& ip=cont.aggidxset.at(gi);
          assert(ip.global()==gi);
          std::size_t column = ip.local();
          cont.sparsity[i].insert(column);

          typedef typename Dune::OwnerOverlapCopyCommunication<int>::OwnerSet OwnerSet;
          if(!OwnerSet::contains(ip.local().attribute()))
            // preserve symmetry for overlap
            cont.sparsity[column].insert(i);
        }
      }
      catch(const Dune::RangeError&) {
        // Entry not present in the new index set. Ignore!
#ifdef DEBUG_REPART
        typedef typename Container::LookupIndexSet GlobalLookup;
        typedef typename GlobalLookup::IndexPair IndexPair;
        typedef typename Dune::OwnerOverlapCopyCommunication<int>::OwnerSet OwnerSet;

        GlobalLookup lookup(cont.aggidxset);
        const IndexPair* pi=lookup.pair(i);
        assert(pi);
        if(OwnerSet::contains(pi->local().attribute())) {
          int rank;
          MPI_Comm_rank(MPI_COMM_WORLD,&rank);
          std::cout<<rank<<cont.aggidxset<<std::endl;
          std::cout<<rank<<": row "<<i<<" (global="<<gi <<") not in index set for owner index "<<pi->global()<<std::endl;
          throw;
        }
#endif
      }
    }

  };
  template<class M, class I>
  typename MatrixSparsityPatternGatherScatter<M,I>::ColIter MatrixSparsityPatternGatherScatter<M,I>::col;

  template<class M, class I>
  typename MatrixSparsityPatternGatherScatter<M,I>::GlobalIndex MatrixSparsityPatternGatherScatter<M,I>::numlimits;


  template<class M, class I>
  struct MatrixRowGatherScatter
  {
    typedef typename I::GlobalIndex GlobalIndex;
    typedef CommMatrixRow<M,I> Container;
    typedef typename M::ConstColIterator ColIter;
    typedef typename std::pair<GlobalIndex,typename M::block_type> Data;
    static ColIter col;
    static Data datastore;
    static GlobalIndex numlimits;

    static const Data& gather(const Container& cont, std::size_t i, std::size_t j)
    {
      if(j==0)
        col=cont.matrix[i].begin();
      else if (col!=cont.matrix[i].end())
        ++col;
      // copy communication: different row sizes for copy rows with the same global index
      // are possible. If all values of current  matrix row are sent, send
      // std::numeric_limits<GlobalIndex>::max()
      // and receiver will ignore it.
      if (col==cont.matrix[i].end()) {
        numlimits = std::numeric_limits<GlobalIndex>::max();
        datastore = Data(numlimits,*col);
        return datastore;
      }
      else {
        // convert local column index to global index
        const typename I::IndexPair* index=cont.idxset.pair(col.index());
        assert(index);
        // Store the data to prevent reference to temporary
        // Only send index if col is no ghost
        if ( index->local().attribute() != 2)
          datastore = Data(index->global(),*col);
        else {
          numlimits = std::numeric_limits<GlobalIndex>::max();
          datastore = Data(numlimits,*col);
        }
        return datastore;
      }
    }
    static void scatter(Container& cont, const Data& data, std::size_t i, [[maybe_unused]] std::size_t j)
    {
      try{
        if (data.first != std::numeric_limits<GlobalIndex>::max()) {
          typename M::size_type column=cont.aggidxset.at(data.first).local();
          cont.matrix[i][column]=data.second;
        }
      }
      catch(const Dune::RangeError&) {
        // This an overlap row and might therefore lack some entries!
      }

    }
  };

  template<class M, class I>
  typename MatrixRowGatherScatter<M,I>::ColIter MatrixRowGatherScatter<M,I>::col;

  template<class M, class I>
  typename MatrixRowGatherScatter<M,I>::Data MatrixRowGatherScatter<M,I>::datastore;

  template<class M, class I>
  typename MatrixRowGatherScatter<M,I>::GlobalIndex MatrixRowGatherScatter<M,I>::numlimits;

  template<typename M, typename C>
  void redistributeSparsityPattern(M& origMatrix, M& newMatrix, C& origComm, C& newComm,
                                   RedistributeInformation<C>& ri)
  {
    typename C::CopySet copyflags;
    typename C::OwnerSet ownerflags;
    typedef typename C::ParallelIndexSet IndexSet;
    typedef RedistributeInformation<C> RI;
    std::vector<typename M::size_type> rowsize(newComm.indexSet().size(), 0);
    std::vector<typename M::size_type> copyrowsize(newComm.indexSet().size(), 0);
    std::vector<typename M::size_type> backwardscopyrowsize(origComm.indexSet().size(), 0);

    // get owner rowsizes
    CommMatrixRowSize<M,RI> commRowSize(origMatrix, ri);
    ri.template redistribute<MatrixRowSizeGatherScatter<M,IndexSet,RI> >(commRowSize,commRowSize);

    origComm.buildGlobalLookup();

    for (std::size_t i=0; i < newComm.indexSet().size(); i++) {
      rowsize[i] = ri.getRowSize(i);
    }
    // get sparsity pattern from owner rows
    CommMatrixSparsityPattern<M,IndexSet>
    origsp(origMatrix, origComm.globalLookup(), newComm.indexSet());
    CommMatrixSparsityPattern<M,IndexSet>
    newsp(origMatrix, origComm.globalLookup(), newComm.indexSet(), rowsize);

    ri.template redistribute<MatrixSparsityPatternGatherScatter<M,IndexSet> >(origsp,newsp);

    // build copy to owner interface to get missing matrix values for novlp case
    if (SolverCategory::category(origComm) == SolverCategory::nonoverlapping) {
      RemoteIndices<IndexSet> *ris = new RemoteIndices<IndexSet>(origComm.indexSet(),
                                                                 newComm.indexSet(),
                                                                 origComm.communicator());
      ris->template rebuild<true>();

      ri.getInterface().free();
      ri.getInterface().build(*ris,copyflags,ownerflags);

      // get copy rowsizes
      CommMatrixRowSize<M,RI> commRowSize_copy(origMatrix, ri);
      ri.template redistribute<MatrixCopyRowSizeGatherScatter<M,IndexSet,RI> >(commRowSize_copy,
                                                                               commRowSize_copy);

      for (std::size_t i=0; i < newComm.indexSet().size(); i++) {
        copyrowsize[i] = ri.getCopyRowSize(i);
      }
      //get copy rowsizes for sender
      ri.redistributeBackward(backwardscopyrowsize,copyrowsize);
      for (std::size_t i=0; i < origComm.indexSet().size(); i++) {
        ri.getBackwardsCopyRowSize(i) = backwardscopyrowsize[i];
      }

      // get sparsity pattern from copy rows
      CommMatrixSparsityPattern<M,IndexSet> origsp_copy(origMatrix,
                                                        origComm.globalLookup(),
                                                        newComm.indexSet(),
                                                        backwardscopyrowsize);
      CommMatrixSparsityPattern<M,IndexSet> newsp_copy(origMatrix, origComm.globalLookup(),
                                                       newComm.indexSet(), copyrowsize);
      ri.template redistribute<MatrixSparsityPatternGatherScatter<M,IndexSet> >(origsp_copy,
                                                                                newsp_copy);

      newsp.completeSparsityPattern(newsp_copy.sparsity);
      newsp.storeSparsityPattern(newMatrix);
    }
    else
      newsp.storeSparsityPattern(newMatrix);

#ifdef DUNE_ISTL_WITH_CHECKING
    // Check for symmetry
    int ret=0;
    typedef typename M::ConstRowIterator RIter;
    for(RIter row=newMatrix.begin(), rend=newMatrix.end(); row != rend; ++row) {
      typedef typename M::ConstColIterator CIter;
      for(CIter col=row->begin(), cend=row->end(); col!=cend; ++col)
      {
        try{
          newMatrix[col.index()][row.index()];
        }catch(const Dune::ISTLError&) {
          std::cerr<<newComm.communicator().rank()<<": entry ("
                   <<col.index()<<","<<row.index()<<") missing! for symmetry!"<<std::endl;
          ret=1;

        }

      }
    }

    if(ret)
      DUNE_THROW(ISTLError, "Matrix not symmetric!");
#endif
  }

  template<typename M, typename C>
  void redistributeMatrixEntries(M& origMatrix, M& newMatrix, C& origComm, C& newComm,
                                 RedistributeInformation<C>& ri)
  {
    typedef typename C::ParallelIndexSet IndexSet;
    typename C::OwnerSet ownerflags;
    std::vector<typename M::size_type> rowsize(newComm.indexSet().size(), 0);
    std::vector<typename M::size_type> copyrowsize(newComm.indexSet().size(), 0);
    std::vector<typename M::size_type> backwardscopyrowsize(origComm.indexSet().size(), 0);

    for (std::size_t i=0; i < newComm.indexSet().size(); i++) {
      rowsize[i] = ri.getRowSize(i);
      if (SolverCategory::category(origComm) == SolverCategory::nonoverlapping) {
        copyrowsize[i] = ri.getCopyRowSize(i);
      }
    }

    for (std::size_t i=0; i < origComm.indexSet().size(); i++)
      if (SolverCategory::category(origComm) == SolverCategory::nonoverlapping)
        backwardscopyrowsize[i] = ri.getBackwardsCopyRowSize(i);


    if (SolverCategory::category(origComm) == SolverCategory::nonoverlapping) {
      // fill sparsity pattern from copy rows
      CommMatrixRow<M,IndexSet> origrow_copy(origMatrix, origComm.globalLookup(),
                                             newComm.indexSet(), backwardscopyrowsize);
      CommMatrixRow<M,IndexSet> newrow_copy(newMatrix, origComm.globalLookup(),
                                            newComm.indexSet(),copyrowsize);
      ri.template redistribute<MatrixRowGatherScatter<M,IndexSet> >(origrow_copy,
                                                                    newrow_copy);
      ri.getInterface().free();
      RemoteIndices<IndexSet> *ris = new RemoteIndices<IndexSet>(origComm.indexSet(),
                                                                 newComm.indexSet(),
                                                                 origComm.communicator());
      ris->template rebuild<true>();
      ri.getInterface().build(*ris,ownerflags,ownerflags);
    }

    CommMatrixRow<M,IndexSet>
    origrow(origMatrix, origComm.globalLookup(), newComm.indexSet());
    CommMatrixRow<M,IndexSet>
    newrow(newMatrix, origComm.globalLookup(), newComm.indexSet(),rowsize);
    ri.template redistribute<MatrixRowGatherScatter<M,IndexSet> >(origrow,newrow);
    if (SolverCategory::category(origComm) != static_cast<int>(SolverCategory::nonoverlapping))
      newrow.setOverlapRowsToDirichlet();
  }

  /**
   * @brief Redistribute a matrix according to given domain decompositions.
   *
   * All the parameters for this function can be obtained by calling
   * graphRepartition with the graph of the original matrix.
   *
   * @param origMatrix The matrix on the original partitioning.
   * @param newMatrix An empty matrix to store the new redistributed matrix in.
   * @param origComm The parallel information of the original partitioning.
   * @param newComm The parallel information of the new partitioning.
   * @param ri The remote index information between the original and the new partitioning.
   * Upon exit of this method it will be prepared for copying from owner to owner vertices
   * for data redistribution.
   * @tparam M The matrix type. It is assumed to be sparse. E.g. BCRSMatrix.
   * @tparam C The type of the parallel information, see OwnerOverlapCopyCommunication.
   */
  template<typename M, typename C>
  void redistributeMatrix(M& origMatrix, M& newMatrix, C& origComm, C& newComm,
                          RedistributeInformation<C>& ri)
  {
    ri.setNoRows(newComm.indexSet().size());
    ri.setNoCopyRows(newComm.indexSet().size());
    ri.setNoBackwardsCopyRows(origComm.indexSet().size());
    redistributeSparsityPattern(origMatrix, newMatrix, origComm, newComm, ri);
    redistributeMatrixEntries(origMatrix, newMatrix, origComm, newComm, ri);
  }
#endif

template<typename M>
  void redistributeMatrixEntries(M& origMatrix, M& newMatrix,
                                 Dune::Amg::SequentialInformation& origComm,
                                 Dune::Amg::SequentialInformation& newComm,
                                 RedistributeInformation<Dune::Amg::SequentialInformation>& ri)
  {
    DUNE_THROW(InvalidStateException, "Trying to redistribute in sequential program!");
  }
  template<typename M>
  void redistributeMatrix(M& origMatrix, M& newMatrix,
                          Dune::Amg::SequentialInformation& origComm,
                          Dune::Amg::SequentialInformation& newComm,
                          RedistributeInformation<Dune::Amg::SequentialInformation>& ri)
  {
    DUNE_THROW(InvalidStateException, "Trying to redistribute in sequential program!");
  }
}
#endif
